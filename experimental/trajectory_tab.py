"""Arrival-confirmed trajectory tracking with default 2 mm waypoint spacing."""

from dataclasses import replace
import time
import tkinter as tk
from tkinter import ttk
import threading

import numpy as np

from arrival_control import ArrivalGate
from command_tab import CommandTab
from command_sequences import XYStep, execute_xy_sequence, project_sequence, validate_projected_points
from swarm_cargo import CargoPositionFeedback
from point_control import hardware_case_name
from control_workflow import ForcePIDFeedback, OptimizerXYFeedback
from direction_xy_control import DirectionWeightedPID, DirectionWeightedXYFeedback
from functions_main import MagneticForcePID
from incremental_xy_control import IncrementalXYFeedback
from assistant_xy_control import AssistantXYFeedback
from incremental_pid_xy_control import IncrementalPIDXYFeedback
from pathlib import Path
from hover_hint import HoverHint


def trajectory_waypoints(start_xy_m, step, spacing_m=.002):
    start, end = np.asarray(start_xy_m, float), np.asarray(step.target_xy_m, float)
    if (start.shape != (2,) or end.shape != (2,) or not np.isfinite([start, end]).all()
            or not np.isfinite(spacing_m) or not 0 < spacing_m <= .005):
        raise ValueError('Trajectory coordinates must be finite XY pairs; waypoint spacing must be at most 5 mm.')
    count = max(1, int(np.ceil(np.linalg.norm(end-start) / spacing_m - 1e-12)))
    return tuple(replace(step, target_xy_m=tuple(end if index == count else start+(end-start)*index/count),
                         wait_seconds=step.wait_seconds if index == count else 0.)
                 for index in range(1, count+1))


class TrajectoryArrivalGate(ArrivalGate):
    """Advance only after position AND measured speed satisfy the full hold."""

    def __init__(self, *args, max_speed_mm_s=.5, velocity_tau_s=.15, **kwargs):
        super().__init__(*args, **kwargs)
        if not np.isfinite(max_speed_mm_s) or max_speed_mm_s <= 0:
            raise ValueError('Waypoint settling speed must be finite and positive.')
        self.max_speed_mm_s = float(max_speed_mm_s)
        self.motion = MagneticForcePID(ki=0., velocity_tau_s=velocity_tau_s)
        self.speed_mm_s = None

    def _reset_motion(self):
        self.motion.reset()
        self.speed_mm_s = None

    def prepare_point(self, **kwargs):
        with self.condition:
            super().prepare_point(**kwargs)
            self._reset_motion()

    def arm(self, target_xy_m, **kwargs):
        with self.condition:
            super().arm(target_xy_m, **kwargs)
            self._reset_motion()

    def _ready(self):
        return (self.speed_mm_s is not None and self.speed_mm_s <= self.max_speed_mm_s
                and super()._ready())

    def observe(self, positions_xy_m, observed_at=None):
        with self.condition:
            previous = self.observed_at
            super().observe(positions_xy_m, observed_at)
            if self.positions is None:
                self._reset_motion()
                return
            if self.observed_at is None or self.observed_at == previous:
                return  # Duplicate/pre-arm frames must not extend the hold.
            self.motion.observe(self.positions.mean(axis=0), self.observed_at)
            self.speed_mm_s = (float(np.linalg.norm(self.motion.velocity)*1000)
                               if self.motion.velocity_ready else None)
            if self.speed_mm_s is None or self.speed_mm_s > self.max_speed_mm_s:
                self.inside_since = None
                self.waiting_for_next = self.next_requested = False
            if self._ready():
                self.waiting_for_next = self.next_requested = True
            self.condition.notify_all()


class TrajectoryTab(CommandTab):
    view_title = 'Trajectory Track'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._center_state = None
        self.center_button = tk.Button(self.next_button.master, text='Center', command=self.move_to_center)
        self.center_button.pack(side='left', before=self.next_button, padx=3)
        self.skip_waypoint_button = tk.Button(self.next_button.master, text='Skip waypoint',
            command=self.skip_waypoint, state='disabled')
        self.skip_waypoint_button.pack(side='left', before=self.next_button, padx=3)
        HoverHint(self.skip_waypoint_button, 'Skip only the active automatic waypoint, even outside arrival tolerance. '
                  'Bypasses its arrival, settling and row wait; the next waypoint uses normal checks. '
                  'Skipping the final waypoint finishes the route.')
        # Prioritize removal of persistent position bias; verify gains on the actual plant.
        for key, value in (('Kp', '.25'), ('Ki', '.02'), ('Kv', '.25'), ('I max µN', '10'), ('Rate deg/s', '0')):
            self.force_pid_settings[key].set(value)
        self.arrival_tolerance_var.set('1.5')
        self.arrival_hold_var.set('1')
        self.force_hold_speed_var.set('.5')
        self.force_step_var.set('90')
        self.cargo_gain_var.set('.15')
        self.cargo_limit_var.set('2.5')
        self.use_learned_var.set(False)
        self.control_requested = True
        self.control_button.config(text='Control on')
        self.refresh_controller_controls()
        self.status.config(text='Trajectory: moves are split into waypoints at most 2 mm apart. '
                           'Each waypoint requires low speed and arrival for the full hold before automatic advance.')

    def control_positions(self):
        positions = super().control_positions()
        # Robot-group trajectories follow the centroid; Cargo mode follows cargo.
        return None if positions is None else positions.mean(axis=0, keepdims=True)

    def build_controller_controls(self, parent):
        self.controller_var = tk.StringVar(value='Incremental XY PID')
        self.incremental_pid_settings = {key: tk.StringVar(value=value) for key, value in (
            ('Command rate mm/s', '.5'), ('Command offset max mm', '100'), ('Slowdown mm', '3'))}
        self.direction_settings = {key: tk.StringVar(value=value) for key, value in (
            ('Sideways integral weight', '.2'), ('Opposing decay s', '2'))}
        self.speed_settings = {key: tk.StringVar(value=value) for key, value in (
            ('Speed limit mm/s', '.5'), ('Brake gain s', '1.5'), ('Max brake mm', '2'))}
        self.incremental_settings = {key: tk.StringVar(value=value) for key, value in (
            ('Step mm', '1'), ('Rate mm/s', '.5'), ('Settle speed mm/s', '.2'),
            ('Settle hold s', '.75'), ('Max offset mm', '5'), ('Step timeout s', '10'))}
        bar = tk.LabelFrame(parent, text='Trajectory controller — choose before running')
        bar.pack(fill='x', padx=8, pady=2)
        controls = tk.Frame(bar)
        controls.pack(fill='x')
        self.controller_selector = ttk.Combobox(controls, textvariable=self.controller_var, state='readonly',
            values=('Incremental XY PID', 'XY PID (current)', 'Direction-weighted XY PID', 'Incremental move-and-settle', 'Online assistant'), width=27)
        widgets = [self.controller_selector]
        self.controller_selector.bind('<<ComboboxSelected>>', lambda event: self.refresh_controller_controls())
        self.incremental_entries = []
        self.direction_entries = []
        self.speed_entries = []
        self.incremental_pid_entries = []
        for key, variable in self.incremental_pid_settings.items():
            cell = tk.Frame(controls)
            tk.Label(cell, text=key).pack(side='left', padx=(4, 2))
            entry = tk.Entry(cell, textvariable=variable, width=5)
            entry.pack(side='left')
            widgets.append(cell)
            self.incremental_pid_entries.append(entry)
        for key, variable in self.speed_settings.items():
            cell = tk.Frame(controls)
            tk.Label(cell, text=key).pack(side='left', padx=(4, 2))
            entry = tk.Entry(cell, textvariable=variable, width=5)
            entry.pack(side='left')
            widgets.append(cell)
            self.speed_entries.append(entry)
        for key, variable in self.direction_settings.items():
            cell = tk.Frame(controls)
            tk.Label(cell, text=key).pack(side='left', padx=(4, 2))
            entry = tk.Entry(cell, textvariable=variable, width=5)
            entry.pack(side='left')
            widgets.append(cell)
            self.direction_entries.append(entry)
        for key, variable in self.incremental_settings.items():
            cell = tk.Frame(controls)
            tk.Label(cell, text=key).pack(side='left', padx=(4, 2))
            entry = tk.Entry(cell, textvariable=variable, width=5)
            entry.pack(side='left')
            widgets.append(cell)
            self.incremental_entries.append(entry)
        self.wrap_annotation_controls(controls, widgets)
        HoverHint(bar, 'Incremental XY PID adjusts the sent optimizer command using PID differences. '
                  'Command rate bounds XY command speed; Command offset max is a radial bound from the waypoint. '
                  'Measured speed cap decreases inside Slowdown mm (to 10% of Speed limit). '
                  'I max is unused in this mode; limits discard excess increments, preventing windup. '
                  'Speed feedback cannot guarantee zero overshoot. '
                  'Automatic controllers prioritize braking above Speed limit, resuming below 80% of the limit. '
                  'Brake gain converts measured velocity to an opposite offset from the measured center, '
                  'bounded by Max brake mm; integral is frozen while braking. Physical speed may overshoot '
                  'between camera samples or when magnetic authority is limited. '
                  'XY PID preserves the current controller. Direction-weighted XY PID reduces sideways '
                  'integral correction (weight 0 to 1) and exponentially unwinds opposing integral '
                  'with the selected decay time. Velocity damping is retained. Incremental ramps a command by up to Step mm '
                  '(maximum 1 mm total XY distance), then waits for low measured speed for Settle hold s. '
                  'Max offset bounds command lead over measurement and extra offset from the waypoint. '
                  'Stops on optimizer limits, a step timeout, or three settled steps with too little progress. '
                  'Period, velocity filter and motor deadband remain shared; PID gains are unused in Incremental.')

    def refresh_controller_controls(self):
        if not hasattr(self, 'controller_selector'):
            return
        busy = self.running or self.precalculating
        manual = hasattr(self, 'manual_mode_var') and self.manual_mode_var.get()
        incremental = self.controller_var.get() in ('Incremental move-and-settle', 'Online assistant')
        self.controller_selector.config(state='disabled' if busy or manual else 'readonly')
        incremental_pid = self.controller_var.get() == 'Incremental XY PID'
        for entry in getattr(self, 'incremental_pid_entries', []):
            entry.config(state='normal' if incremental_pid and not busy and not manual else 'disabled')
        for entry in self.incremental_entries:
            entry.config(state='normal' if incremental and not busy and not manual else 'disabled')
        for entry in getattr(self, 'direction_entries', []):
            entry.config(state='normal' if self.controller_var.get() == 'Direction-weighted XY PID'
                         and not busy and not manual else 'disabled')
        for entry in getattr(self, 'speed_entries', []):
            entry.config(state='normal' if not busy and not manual else 'disabled')
        for key in ('Kp', 'Ki', 'Kv', 'I max µN'):
            entry = getattr(self, 'force_pid_entries', {}).get(key)
            if entry is not None:
                entry.config(state='normal' if not incremental and not busy and not manual
                             and not (incremental_pid and key.startswith('I max')) else 'disabled')

    def path_shortcuts(self):
        return (('Center', self.move_to_center, self.center_button),
                ('Skip waypoint', self.skip_waypoint, self.skip_waypoint_button))

    def skip_waypoint(self):
        gate = getattr(self, 'arrival_gate', None)
        if (not self.running or getattr(self, 'run_manual_mode', False) or gate is None
                or self.stop_event.is_set() or getattr(self, '_center_state', None) is not None):
            return
        if gate.request_skip():
            feedback = getattr(self, 'feedback', None)
            if feedback is not None:
                feedback.publish(waypoints_skipped=gate.skip_count)
            self.skip_waypoint_button.config(state='disabled')
            self.status.config(text='Skipping the current waypoint by request; normal arrival checks resume at the next waypoint.')

    def move_to_center(self):
        if self.running or self.precalculating or not self.operation_allowed():
            self.status.config(text='Wait for the current operation to finish before moving to Center.')
            return
        selection = self.table.selection()
        index = int(selection[0]) if selection else 0
        zero = self.steps[index].zero_angle_deg if self.steps else 180.
        self._center_state = dict(steps=self.steps, pixels=self.pixels, sequence_path=self.sequence_path,
            input_units=self.input_units, selection=selection, label=self.file_label.cget('text'),
            rows=[(row, self.table.item(row)) for row in self.table.get_children()],
            corrections=dict(self.display_corrections), ratio=self.point_ratio_var.get(), angle=self.point_angle_var.get())
        self.steps = [XYStep((0., 0.), 0., zero, ratio=1., angle_deg=None)]
        self.sequence_path, self.input_units, self.pixels, self.active_index = None, 'm', None, None
        self.display_corrections = {}
        self.table.delete(*self.table.get_children())
        self.table.insert('', 'end', iid='0', text='Center',
            values=('0', '0', '0.00', '0.00', '0.00', '0.00', '0', f'{zero:g}', '1', 'Auto', 'Pending', ''))
        self.table.selection_set('0')
        self.point_ratio_var.set('1')
        self.point_angle_var.set('Auto')
        self.file_label.config(text='Center reset — direct XY (0, 0) mm, ratio 1')
        try:
            self.run_selected()
        finally:
            if not self.running:
                self._restore_center_state()

    def _restore_center_state(self):
        state, self._center_state = self._center_state, None
        if state is None:
            return
        self.steps, self.pixels = state['steps'], state['pixels']
        self.sequence_path, self.input_units = state['sequence_path'], state['input_units']
        self.display_corrections = state['corrections']
        self.active_index = self.trajectory_waypoint = None
        self.table.delete(*self.table.get_children())
        for row, item in state['rows']:
            self.table.insert('', 'end', iid=row, **item)
        if state['selection']:
            self.table.selection_set(state['selection'])
        self.file_label.config(text=state['label'])
        self.point_ratio_var.set(state['ratio'])
        self.point_angle_var.set(state['angle'])
        self.refresh_next_button()
        self.render()

    def make_arrival_gate(self):
        if getattr(self, '_center_state', None) is not None:
            return ArrivalGate(require_arrival=False)
        if hasattr(self, 'manual_mode_var') and self.manual_mode_var.get():
            return ArrivalGate(float(self.arrival_tolerance_var.get()), float(self.arrival_hold_var.get()),
                               require_arrival=False)
        tolerance = float(self.arrival_tolerance_var.get())
        return TrajectoryArrivalGate(tolerance, float(self.arrival_hold_var.get()), require_arrival=True,
            max_speed_mm_s=float(self.force_hold_speed_var.get()),
            velocity_tau_s=float(self.force_pid_settings['Velocity filter s'].get()))

    def configure_run_feedback(self, feedback):
        if getattr(self, '_center_state', None) is not None:
            return None
        if hasattr(self, 'manual_mode_var') and self.manual_mode_var.get():
            return ForcePIDFeedback(validate=feedback.validate, manual=True, pid=MagneticForcePID(ki=0.))
        mode = self.controller_var.get() if hasattr(self, 'controller_var') else 'XY PID (current)'
        if mode not in ('Incremental XY PID', 'XY PID (current)', 'Direction-weighted XY PID', 'Incremental move-and-settle', 'Online assistant'):
            raise ValueError('Select a supported trajectory controller.')
        common = ('Period s', 'Velocity filter s', 'Deadband deg')
        incremental = mode in ('Incremental move-and-settle', 'Online assistant')
        keys = common if incremental else tuple(self.force_pid_settings)
        if mode == 'Incremental XY PID':
            keys = (*common, 'Kp', 'Ki', 'Kv')
        values = {key: float(self.force_pid_settings[key].get()) for key in keys}
        settings = getattr(self, 'speed_settings', {})
        speed = dict(speed_limit_mm_s=float(settings['Speed limit mm/s'].get()) if settings else .5,
                     brake_gain_s=float(settings['Brake gain s'].get()) if settings else 1.5,
                     max_brake_mm=float(settings['Max brake mm'].get()) if settings else 2.)
        if mode == 'Incremental XY PID':
            settings = {key: float(var.get()) for key, var in self.incremental_pid_settings.items()}
            result = IncrementalPIDXYFeedback(validate=feedback.validate, **speed,
                command_rate_mm_s=settings['Command rate mm/s'],
                command_offset_mm=settings['Command offset max mm'], slowdown_mm=settings['Slowdown mm'],
                pid=MagneticForcePID(kp=values['Kp'], ki=values['Ki'], kv=values['Kv'],
                    velocity_tau_s=values['Velocity filter s'], period_s=values['Period s']),
                max_angle_step_deg=90., angle_rate_deg_s=0., preview_only=self.force_preview_var.get(),
                motor_deadband_deg=values['Deadband deg'], hold_speed_mm_s=float(self.force_hold_speed_var.get()))
            if self.control_requested:
                result.enabled.set()
            return result
        if incremental:
            settings = {key: float(var.get()) for key, var in self.incremental_settings.items()}
            controller = AssistantXYFeedback if mode == 'Online assistant' else IncrementalXYFeedback
            extra = {'bridge_directory': Path(__file__).resolve().parent/'live_control'} if mode == 'Online assistant' else {}
            result = controller(validate=feedback.validate, **extra, **speed,
                pid=MagneticForcePID(kp=0., ki=0., kv=0., velocity_tau_s=values['Velocity filter s'],
                                     period_s=values['Period s']),
                step_mm=settings['Step mm'], rate_mm_s=settings['Rate mm/s'],
                settle_speed_mm_s=settings['Settle speed mm/s'], settle_seconds=settings['Settle hold s'],
                max_bias_mm=settings['Max offset mm'], timeout_seconds=settings['Step timeout s'],
                max_angle_step_deg=90., angle_rate_deg_s=0.,
                preview_only=self.force_preview_var.get(), motor_deadband_deg=values['Deadband deg'],
                hold_speed_mm_s=float(self.force_hold_speed_var.get()))
            if self.control_requested:
                result.enabled.set()
            return result
        controller, pid_class, extra = OptimizerXYFeedback, MagneticForcePID, {}
        if mode == 'Direction-weighted XY PID':
            controller, pid_class = DirectionWeightedXYFeedback, DirectionWeightedPID
            extra = dict(sideways_weight=float(self.direction_settings['Sideways integral weight'].get()),
                         opposing_decay_s=float(self.direction_settings['Opposing decay s'].get()))
        result = controller(validate=feedback.validate, **speed,
            pid=pid_class(**extra, kp=values['Kp'], ki=values['Ki'], kv=values['Kv'],
                velocity_tau_s=values['Velocity filter s'], integral_limit_n=values['I max µN']*.001,
                period_s=values['Period s']), max_angle_step_deg=90., angle_rate_deg_s=0.,
            preview_only=self.force_preview_var.get(), motor_deadband_deg=values['Deadband deg'],
            hold_speed_mm_s=float(self.force_hold_speed_var.get()))
        if self.control_requested:
            result.enabled.set()
        return result

    def robot_allocation_position(self):
        xy = self.tracking_group_xy
        return None if xy is None else np.asarray(xy, float)*.01

    def diagnostic_settings(self):
        settings = super().diagnostic_settings()
        settings['pid_display_units'] = {
            {'Kv': 'Kd', 'I max µN': 'I max mm'}.get(key, key): variable.get()
            for key, variable in self.force_pid_settings.items() if key != 'Rate deg/s'}
        settings.update(workspace=self.view_title, controller='anchored_xy_optimizer_pid',
                        arrival_rule='position_and_speed_for_full_hold',
                        integral_carry_between_points=True, integral_carry_between_runs=False,
                        pid_units=dict(Kp='mm/mm', Ki='1/s', Kv='s', integral_limit='mm'),
                        initial_pose='last_sent_angles_and_measured_xy')
        if hasattr(self, 'controller_var'):
            settings['controller_selection'] = self.controller_var.get()
            settings['speed_settings'] = {key: variable.get() for key, variable in self.speed_settings.items()}
            settings['speed_braking_rule'] = 'brake_above_limit_resume_at_80_percent'
            settings['incremental_settings'] = {key: variable.get() for key, variable in self.incremental_settings.items()}
            if self.controller_var.get() == 'Incremental XY PID':
                settings.update(controller=IncrementalPIDXYFeedback.control_mode,
                    incremental_pid_settings={key: var.get() for key, var in self.incremental_pid_settings.items()},
                    integral_limit_used=False, integral_carry_between_points=False,
                    command_carry_between_points=True,
                    speed_braking_rule='distance_scaled_cap_brake_above_resume_at_80_percent',
                    pid_units=dict(Kp='mm/mm', Ki='1/s', Kv='s'))
            elif self.controller_var.get() == 'Direction-weighted XY PID':
                settings.update(controller=DirectionWeightedXYFeedback.control_mode,
                                direction_settings={key: variable.get() for key, variable in self.direction_settings.items()})
            elif self.controller_var.get() in ('Incremental move-and-settle', 'Online assistant'):
                controller = AssistantXYFeedback if self.controller_var.get() == 'Online assistant' else IncrementalXYFeedback
                settings.update(controller=controller.control_mode,
                                integral_carry_between_points=False, pid_units=None)
        if hasattr(self, 'manual_mode_var') and self.manual_mode_var.get():
            settings.update(controller='manual_xy', arrival_rule='manual_next', integral_carry_between_points=False)
        return settings

    def change_manual_mode(self):
        if self.running:
            self.manual_mode_var.set(getattr(self, 'run_manual_mode', False))
            self.status.config(text='Stop the current run before switching between Manual and automatic trajectory control.')
            return
        super().change_manual_mode()
        self.refresh_controller_controls()

    def run(self, *, selected_only=False):
        if self.running:
            return
        if getattr(self, '_center_state', None) is not None:
            self._run_center_reset()
            return
        if self.manual_mode_var.get():
            self.trajectory_waypoint = None
            super().run(selected_only=selected_only)
            return
        if not self.control_requested or self.force_preview_var.get():
            self.status.config(text='Automatic trajectory tracking requires Control on and Preview only off.')
            return
        if self.tracking_tracker is None:
            self.status.config(text='Detect the robot group or cargo before starting a trajectory.')
            return
        positions = self.control_positions()
        received = getattr(self, 'measurement_received_at', None)
        if positions is None or received is None or time.monotonic()-received > 1.:
            self.status.config(text='Wait for a fresh robot-group or cargo measurement before starting.')
            return
        try:
            selected = self.table.selection() if selected_only else ()
            step = self.steps[int(selected[0]) if selected else 0]
            self.commander.applied_pose(self.calibration.control_case_var.get(), step.zero_angle_deg)
        except (ValueError, IndexError) as error:
            self.status.config(text=str(error))
            return
        self.trajectory_start_xy_m = tuple(positions[0])
        self.trajectory_waypoint = None
        super().run(selected_only=selected_only)

    def _run_center_reset(self):
        # Hardware reset deliberately has no camera, calibration, or tracking prerequisites.
        try:
            case = hardware_case_name(self.calibration.control_case_var.get())
            if not case:
                raise ValueError('Select the hardware case first.')
            port = self.commander.port_provider()
            if port is None or not getattr(port, 'is_open', False):
                raise ValueError('Connect the serial controller first.')
        except (ValueError, TypeError) as error:
            self.status.config(text=f'Center reset unavailable: {error}')
            return
        self.running, self.run_manual_mode = True, False
        self.arrival_gate = self.feedback = None
        self.run_indices = (0,)
        self.active_index = None
        self.stop_event.clear()
        self.job += 1
        job = self.job
        step = self.steps[0]
        self.set_running(True)
        for widget in (self.load_button, self.run_button, self.run_selected_button, self.precalculate_button):
            widget.config(state='disabled')
        self.stop_button.config(state='normal')
        self.refresh_next_button()
        self.status.config(text='Resetting to Center: direct XY (0, 0), ratio 1. Detection is not required.')

        def active():
            return (not self.stop_event.is_set() and self.job == job
                    and self.commander.port_provider() is port and getattr(port, 'is_open', False))

        def worker():
            try:
                self.events.put((job, 'calculating', 0, None))
                result = self.commander.move(case, (0., 0.), ratio=1., angle_deg=None,
                    settle_seconds=0., zero_angle_deg=step.zero_angle_deg, active=active,
                    on_sent=lambda value: self.events.put((job, 'sent', 0, value)))
                self.events.put((job, 'finished' if result is not None and active() else 'stopped', None, None))
            except Exception as error:
                self.events.put((job, 'error', None, str(error)))

        threading.Thread(target=worker, daemon=True).start()
        self.after_id = self.parent.after(50, self.poll)

    def precalculate_all(self):
        self.status.config(text='Trajectory waypoints are generated from the live position when Run is pressed.')

    def execute_sequence(self, steps, commander, case_name, **options):
        if getattr(self, '_center_state', None) is not None:
            # Reset is a single center pose, independent of trajectory progression.
            reset_options = dict(options, arrival_gate=None, feedback=None, prepared=False,
                                 initial_corrections=[[0., 0.]], on_correction=lambda *args: None)
            return execute_xy_sequence(steps, commander, case_name, **reset_options)
        if getattr(options.get('feedback'), 'manual', False) is True:
            # Preserve row identity and saved/clicked u. Manual runs use Next,
            # rather than automatic interpolated waypoint progression.
            return execute_xy_sequence(steps, commander, case_name, **options)
        start = self.trajectory_start_xy_m
        feedback = options.get('feedback')
        if feedback is not None and getattr(feedback, 'incremental_angles', False) is True:
            zero = steps[0].zero_angle_deg
            if any(step.zero_angle_deg != zero for step in steps):
                raise ValueError('Continuous trajectory tracking requires one motor zero angle throughout the route.')
            pose = commander.applied_pose(case_name, zero)
            if isinstance(feedback, OptimizerXYFeedback):
                feedback.seed_anchor(start, pose)
            else:
                feedback.initial_sent(pose)
                feedback.last_sent_at = pose['sent_time_s']
            feedback.publish(initial_pose='last_sent_angles', commanded_xy_mm=None)
        report, remember = options['report'], options.get('on_correction', lambda index, u: None)
        plans = []
        # Validate the entire interpolated route before sending its first pose.
        for step in steps:
            waypoints = trajectory_waypoints(start, step)
            pixels = project_sequence(waypoints, self.setup, validate=False)
            validate_projected_points(waypoints, self.setup, pixels,
                                      allow_outside_pins=self.run_allow_extrapolation)
            plans.append(waypoints)
            start = step.target_xy_m
        for index, waypoints in enumerate(plans):
            for number, waypoint in enumerate(waypoints, 1):
                def forward(kind, ignored, result):
                    if kind == 'completed':
                        return
                    report(kind, index, result)
                    if kind in ('sent', 'waiting_arrival', 'corrected'):
                        report('trajectory_waypoint', index, dict(target_xy_m=waypoint.target_xy_m,
                                                                  number=number, total=len(waypoints)))
                sub_options = dict(options, report=forward, prepared=False, initial_corrections=[[0., 0.]],
                    on_correction=(lambda ignored, u: remember(index, u)) if number == len(waypoints) else lambda *args: None)
                outcome = execute_xy_sequence((waypoint,), commander, case_name, **sub_options)
                if outcome != 'finished':
                    return outcome
            report('completed', index, None)
        return 'finished'

    def handle_execution_event(self, kind, index, result):
        if kind == 'waypoint_skipped':
            self.status.config(text=f'Point {index+1}: current waypoint skipped by request.')
            recorder = getattr(self, 'diagnostics', None)
            if recorder is not None:
                recorder.record('waypoint_skipped', dict(point=index+1, **result))
            return True
        if kind in ('finished', 'stopped', 'error'):
            feedback = getattr(self, 'feedback', None)
            if isinstance(feedback, AssistantXYFeedback):
                feedback.publish_bridge(stopped=True)
            self.trajectory_waypoint = None
            if getattr(self, '_center_state', None) is not None:
                # Restore after the base handler has updated the temporary Center row.
                self.parent.after_idle(self._restore_center_state)
        if kind != 'trajectory_waypoint':
            return False
        self.active_index = index
        self.trajectory_waypoint = result['target_xy_m']
        x, y = np.asarray(self.trajectory_waypoint) * 1000
        self.status.config(text=f'Trajectory point {index+1}: waypoint {result["number"]}/{result["total"]}, '
                           f'XY ({x:.2f}, {y:.2f}) mm; waiting for settled arrival.')
        self.render()
        return True

    def current_tracking_target(self):
        gate = getattr(self, 'arrival_gate', None)
        if self.running and gate is not None and gate.target is not None:
            return tuple(gate.target)
        return super().current_tracking_target()

    def refresh_next_button(self):
        self.refresh_controller_controls()
        manual_check = getattr(self, 'manual_check', None)
        if manual_check is not None:
            manual_check.config(state='disabled' if self.running or self.precalculating else 'normal')
        center = getattr(self, 'center_button', None)
        skip = getattr(self, 'skip_waypoint_button', None)
        gate = getattr(self, 'arrival_gate', None)
        if skip is not None:
            can_skip = (self.running and not getattr(self, 'run_manual_mode', False) and gate is not None
                        and gate.target is not None and not gate.skip_requested and not self.stop_event.is_set()
                        and getattr(self, '_center_state', None) is None)
            skip.config(state='normal' if can_skip else 'disabled')
        if center is not None:
            center.config(state='disabled' if self.running or self.precalculating else 'normal')
        if self.running and not getattr(self, 'run_manual_mode', False):
            self.next_button.config(state='disabled', text='Automatic waypoints')
        else:
            super().refresh_next_button()
