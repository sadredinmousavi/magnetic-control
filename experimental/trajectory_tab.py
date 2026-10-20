"""Arrival-confirmed trajectory tracking with a maximum 5 mm waypoint spacing."""

from dataclasses import replace
import time

import numpy as np

from arrival_control import ArrivalGate
from command_tab import CommandTab
from command_sequences import execute_xy_sequence, project_sequence, validate_projected_points
from position_control import PositionPID
from swarm_cargo import CargoPositionFeedback


def trajectory_waypoints(start_xy_m, step, spacing_m=.005):
    start, end = np.asarray(start_xy_m, float), np.asarray(step.target_xy_m, float)
    if (start.shape != (2,) or end.shape != (2,) or not np.isfinite([start, end]).all()
            or not np.isfinite(spacing_m) or not 0 < spacing_m <= .005):
        raise ValueError('Trajectory coordinates must be finite XY pairs; waypoint spacing must be at most 5 mm.')
    count = max(1, int(np.ceil(np.linalg.norm(end-start) / spacing_m - 1e-12)))
    return tuple(replace(step, target_xy_m=tuple(end if index == count else start+(end-start)*index/count),
                         wait_seconds=step.wait_seconds if index == count else 0.)
                 for index in range(1, count+1))


class TrajectoryArrivalGate(ArrivalGate):
    """Advance automatically only after fresh observations confirm arrival."""

    def observe(self, positions_xy_m, observed_at=None):
        super().observe(positions_xy_m, observed_at)
        with self.condition:
            if self._ready():
                self.waiting_for_next = self.next_requested = True
                self.condition.notify_all()


class TrajectoryTab(CommandTab):
    view_title = 'Trajectory Track'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Conservative starting gains; damping must be verified on the actual plant.
        for key, value in (('Kp', '.008'), ('Ki', '0'), ('Kv', '.025'), ('Rate deg/s', '10')):
            self.force_pid_settings[key].set(value)
        self.force_step_var.set('5')
        self.cargo_gain_var.set('.15')
        self.cargo_limit_var.set('2.5')
        self.use_learned_var.set(False)
        self.control_requested = True
        self.control_button.config(text='Control on')
        self.status.config(text='Trajectory: moves are split into waypoints at most 5 mm apart. '
                           'Each waypoint requires arrival and the in-range hold before automatic advance.')

    def control_positions(self):
        positions = super().control_positions()
        # Robot-group trajectories follow the centroid; Cargo mode follows cargo.
        return None if positions is None else positions.mean(axis=0, keepdims=True)

    def make_arrival_gate(self):
        tolerance = float(self.arrival_tolerance_var.get())
        if tolerance > 1.:
            raise ValueError('Use arrival tolerance at most 1 mm for the 5 mm trajectory.')
        return TrajectoryArrivalGate(tolerance, float(self.arrival_hold_var.get()), require_arrival=True)

    def configure_run_feedback(self, feedback):
        if isinstance(feedback, CargoPositionFeedback):
            feedback.pid = PositionPID(kp=float(self.cargo_gain_var.get()), ki=0., kd=.3,
                limit_mm=min(2.5, float(self.cargo_limit_var.get())), slew_mm_s=.5, period_s=.25)
        return feedback

    def run(self, *, selected_only=False):
        if self.running:
            return
        if self.manual_mode_var.get() or not self.control_requested or self.force_preview_var.get():
            self.status.config(text='Trajectory tracking requires Control on, Manual off and Preview only off.')
            return
        if self.tracking_tracker is None:
            self.status.config(text='Detect the robot group or cargo before starting a trajectory.')
            return
        positions = self.control_positions()
        received = getattr(self, 'measurement_received_at', None)
        if positions is None or received is None or time.monotonic()-received > 1.:
            self.status.config(text='Wait for a fresh robot-group or cargo measurement before starting.')
            return
        self.trajectory_start_xy_m = tuple(positions[0])
        self.trajectory_waypoint = None
        super().run(selected_only=selected_only)

    def precalculate_all(self):
        self.status.config(text='Trajectory waypoints are generated from the live position when Run is pressed.')

    def execute_sequence(self, steps, commander, case_name, **options):
        start = self.trajectory_start_xy_m
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
        if kind in ('finished', 'stopped', 'error'):
            self.trajectory_waypoint = None
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
        if self.running:
            self.next_button.config(state='disabled', text='Automatic waypoints')
        else:
            super().refresh_next_button()
