"""GUI005 desired XY overlay and explicit, cancellable sequence execution."""

import json
import csv
from collections import deque
from dataclasses import replace
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from command_sequences import (
    XY_SEQUENCE_DIR, read_xy_sequence, resolve_xy_units, project_sequence, execute_xy_sequence,
    point_limits, sequence_diagnostics, validate_projected_points,
    write_xy_sequence,
)
from point_control import hardware_case_name, load_hardware_case
from closed_loop_calibration import MovingROICalibration, default_setup_name
from arrival_control import ArrivalGate
from command_recording import CommandRecording, draw_command_overlay, video_crop_circle
from control_diagnostics import ControlDiagnostics
from camera_popout import CameraPopout, show_view_dialog
from control_workflow import ForcePIDFeedback
from functions_main import MagneticForcePID
from hover_hint import HoverHint
from learned_corrections import LearnedCorrections, correction_context
from axis_calibration import RobotAxes


class CommandTab:
    def __init__(self, parent, calibration, commander, camera_running, *, set_running=lambda value: None,
                 visible=lambda: True):
        self.parent, self.calibration, self.commander = parent, calibration, commander
        self.camera_running, self.set_running, self.visible = camera_running, set_running, visible
        self.steps, self.pixels, self.setup = [], None, None
        self.sequence_path, self.input_units = None, None
        self.frame, self.frame_received_at = None, None
        self.running = False
        self.feedback = None
        self.command_roi = None
        self.roi_override = False
        self.calibration_revision = None
        self.roi_placement_var = tk.BooleanVar(value=False)
        self.control_requested = False
        self.repeat_motors_var = tk.BooleanVar(value=True)
        self.motor_repeat_count_var = tk.StringVar(value='3')
        self.motor_repeat_selectors = []
        self.learned = LearnedCorrections()
        self.latest_applied = LearnedCorrections()
        self.manual_targets = LearnedCorrections()
        self.display_corrections = {}
        self.manual_mode_var = tk.BooleanVar(value=False)
        self.manual_mode_active = False
        self.show_commanded_var = tk.BooleanVar(value=False)
        self.use_learned_var = tk.BooleanVar(value=True)
        self.force_step_var = tk.StringVar(value='10')
        self.force_preview_var = tk.BooleanVar(value=False)
        self.force_pid_settings = {key: tk.StringVar(value=value) for key, value in (
            ('Kp', '.01'), ('Ki', '.0005'), ('Kv', '.01'), ('Period s', '.25'),
            ('Velocity filter s', '.15'), ('Rate deg/s', '20'), ('I max µN', '.5'), ('Deadband deg', '.1'))}
        self.force_hold_speed_var = tk.StringVar(value='.5')
        self.precalculating = False
        self.precalculated_signature = None
        self.preparation_stop = threading.Event()
        self.preparation_events = queue.Queue()
        self.preparation_after = None
        self.stop_event = threading.Event()
        self.events = queue.Queue()
        self.job = 0
        self.active_index = None
        self.after_id = None
        self.arrival_gate = None
        self.cell_editor = None
        self.render_revision = 0
        self.camera_popout = None
        self.recording = None
        self.diagnostics = None
        self.diagnostic_frame = 0
        self.recording_fps_var = tk.StringVar(value='5')
        self.crop_enabled_var = tk.BooleanVar(value=False)
        self.crop_fit_var = tk.BooleanVar(value=True)
        self.crop_radius_var = tk.StringVar(value='300')
        self.crop_center = None
        self.crop_placing = False
        self.arrival_tolerance_var = tk.StringVar(value="1")
        self.arrival_hold_var = tk.StringVar(value="0.5")
        self.tracking_tracker = None
        self.tracking_detections = []
        self.tracking_group_xy = None
        self.tracking_enabled = True
        self.tracking_signature = None
        self.tracking_rows = []
        self.robot_trails = {}
        self.tracking_started_at = None
        self.tracking_frame_number = 0
        self.tracking_count_var = tk.StringVar(value="1")
        self.tracking_area_var = tk.StringVar(value="1")
        self.tracking_contrast_var = tk.StringVar(value="10")
        self.tracking_radius_var = tk.StringVar(value="300")
        self.show_path_var = tk.BooleanVar(value=False)
        self.show_roi_var = tk.BooleanVar(value=False)
        self.show_trails_var = tk.BooleanVar(value=False)
        camera_view_bar = tk.Frame(parent, background='#edf4ff')
        camera_view_bar.pack(fill='x', padx=8, pady=(6, 2))
        tk.Label(camera_view_bar, text='Camera view', background='#edf4ff', foreground='#173c68',
                 font=('Segoe UI', 10, 'bold')).pack(side='left', padx=8)
        self.popout_button = tk.Button(camera_view_bar, text='Pop-out camera', command=self.open_camera_popout,
                                       background='#195da8', foreground='white', activebackground='#12477e',
                                       activeforeground='white', font=('Segoe UI', 10, 'bold'), padx=12)
        self.popout_button.pack(side='left', padx=8, pady=4)
        self.diagnostics_button = tk.Button(camera_view_bar, text='Start diagnostics', command=self.toggle_diagnostics)
        self.diagnostics_button.pack(side='left', padx=4)
        HoverHint(self.diagnostics_button, 'Record each received Commands frame and motor calculation/send events. No video encoding. Stop diagnostics saves files in experimental/diagnostics.')
        tk.Label(camera_view_bar, text='Live processing FPS', background='#edf4ff', foreground='#173c68').pack(side='left')
        tk.Spinbox(camera_view_bar, from_=1, to=15, textvariable=self.calibration.live_fps_var, width=3).pack(side='left', padx=4)
        tk.Label(camera_view_bar, text='Opens the larger camera window; all controls remain here.',
                 background='#edf4ff', foreground='#173c68').pack(side='left', padx=4)
        bar = tk.Frame(parent)
        bar.pack(fill="x", padx=8, pady=6)
        self.load_button = tk.Button(bar, text="Load XY sequence", command=self.load_sequence)
        self.load_button.pack(side="left", padx=3)
        tk.Button(bar, text="Export XY sequence", command=self.export_sequence).pack(side="left", padx=3)
        self.precalculate_button = tk.Button(bar, text='Precalculate all', command=self.precalculate_all)
        self.precalculate_button.pack(side='left', padx=3)
        actions = tk.Frame(parent)
        actions.pack(fill="x", padx=8, pady=3)
        self.run_selected_button = tk.Button(actions, text="Run selected point", command=self.run_selected)
        self.run_selected_button.pack(side="left", padx=3)
        self.run_button = tk.Button(actions, text="Run all", command=self.run)
        self.run_button.pack(side="left", padx=3)
        self.stop_button = tk.Button(actions, text="Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=3)
        self.control_button = tk.Button(actions, text='Control off', command=self.toggle_control)
        self.control_button.pack(side='left', padx=3)
        self.next_button = tk.Button(actions, text="Next point", command=self.next_point, state="disabled")
        self.next_button.pack(side="left", padx=3)
        self.allow_extrapolation_var = tk.BooleanVar(value=True)
        self.extrapolation_check = tk.Checkbutton(actions, text="Allow beyond pins (extrapolated)",
                                                 variable=self.allow_extrapolation_var)
        self.extrapolation_check.pack(side="left", padx=8)
        tk.Label(bar, text="Sequence units").pack(side="left", padx=(12, 3))
        self.units_var = tk.StringVar(value="Auto")
        self.units_combo = ttk.Combobox(bar, textvariable=self.units_var, values=("Auto", "mm", "cm", "m"),
                                        width=5, state="readonly")
        self.units_combo.pack(side="left")
        self.units_combo.bind("<<ComboboxSelected>>", self.change_units)
        self.file_label = tk.Label(parent, text="No XY sequence loaded.", anchor="w")
        self.file_label.pack(fill="x", padx=10)
        self.help_button = tk.Button(bar, text='Help ?')
        self.help_button.pack(side='left', padx=6)
        self.help_hint = HoverHint(self.help_button, (
            "The loaded calibration supplies the default ROI and detection settings. Check Phase 3 robots; "
            "use Place ROI to reposition it if needed. Without calibration, Place ROI still starts pixel detection. "
            "Physical XY and running require a complete Phase 2 calibration. "
            "Calibration sizes: cm. Sequence: select mm, cm or m. Auto reads the unit header; without a header it uses mm. "
            "Table shows mm. Double-click Ratio, Angle, Wait or Zero to edit each point (Enter applies). "
            "Ratio defaults to 1; Angle Auto is unconstrained and is ignored for ratio 1. "
            "Precalculate all prepares motor angles without moving; otherwise each point is calculated on demand. "
            "Next can queue one advance during calculation; arrival is not required. The row's minimum wait still applies. "
            "When idle, Next runs the row after the selected point (First point at the end). "
            "Press Next point to advance (Finish for the last point). There is no automatic progression. "
            "Stop prevents further commands; it does not undo the last motor pose."
        ))
        table_frame = tk.Frame(parent)
        table_frame.pack(fill="x", padx=8)
        self.table = ttk.Treeview(table_frame, columns=("x", "y", "ux", "uy", "cx", "cy", "wait", "zero", "ratio", "angle", "state", "angles"),
                                  show="tree headings", height=6, selectmode="browse")
        self.table.tag_configure("outside", foreground="#b00020")
        self.table.heading("#0", text="Point")
        self.table.column("#0", width=55, stretch=False)
        for key, title, width in (("x", "Desired X", 75), ("y", "Desired Y", 75),
                                  ("ux", "u X", 65), ("uy", "u Y", 65),
                                  ("cx", "Command X", 80), ("cy", "Command Y", 80),
                                  ("wait", "Wait (s)", 65), ("zero", "Zero (deg)", 75),
                                  ("ratio", "Ratio", 60), ("angle", "Angle (deg)", 80),
                                  ("state", "State", 95), ("angles", "Calculated motor angles (deg)", 380)):
            self.table.heading(key, text=title)
            self.table.column(key, width=width, stretch=(key == "angles"))
        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        horizontal = ttk.Scrollbar(table_frame, orient='horizontal', command=self.table.xview)
        self.table.configure(xscrollcommand=horizontal.set)
        horizontal.pack(side='bottom', fill='x')
        self.table.pack(side="left", fill="x", expand=True)
        self.table.bind("<<TreeviewSelect>>", self.select_point)
        self.table.bind("<Double-1>", self.edit_point_cell)
        point_settings = tk.Frame(parent)
        point_settings.pack(fill='x', padx=8, pady=3)
        self.point_ratio_var = tk.StringVar(value='1')
        self.point_angle_var = tk.StringVar(value='Auto')
        tk.Label(point_settings, text='Selected point: Ratio').pack(side='left', padx=3)
        tk.Entry(point_settings, textvariable=self.point_ratio_var, width=6).pack(side='left')
        tk.Label(point_settings, text='Equilibrium angle (deg / Auto)').pack(side='left', padx=6)
        tk.Entry(point_settings, textvariable=self.point_angle_var, width=7).pack(side='left')
        tk.Button(point_settings, text='Apply to selected point', command=self.apply_point_settings).pack(side='left', padx=6)
        self.status = tk.Label(parent, text="Load an XY sequence and complete/load Calibration first.", anchor="w", wraplength=1000)
        self.status.pack(fill="x", padx=10, pady=4)
        self.limits_label = tk.Label(parent, text="", anchor="w", justify="left", wraplength=1000)
        self.limits_label.pack(fill="x", padx=10)
        tracking_bar = tk.Frame(parent)
        tracking_bar.pack(fill="x", padx=8, pady=3)
        self.place_roi_button = tk.Button(tracking_bar, text='Place ROI', command=self.begin_roi_placement)
        self.place_roi_button.pack(side='left', padx=2)
        tk.Button(tracking_bar, text='Clear ROI', command=self.clear_command_roi).pack(side='left', padx=2)
        HoverHint(self.place_roi_button, 'Set Phase 3 robots and ROI radius, then click the camera to place/reposition the moving ROI. No motors move.')
        tk.Button(tracking_bar, text="Export tracking CSV", command=self.export_tracking).pack(side="left", padx=2)
        for label, variable in (("Phase 3 robots", self.tracking_count_var), ("Min area", self.tracking_area_var),
                                ("Contrast", self.tracking_contrast_var), ("ROI radius (px)", self.tracking_radius_var)):
            tk.Label(tracking_bar, text=label).pack(side="left", padx=(7, 2))
            tk.Entry(tracking_bar, textvariable=variable, width=4).pack(side="left")
        options = tk.Frame(parent)
        options.pack(fill="x", padx=8)
        for label, variable in (("Show desired lines", self.show_path_var), ("Show moving ROI", self.show_roi_var),
                                ('Show commanded point', self.show_commanded_var),
                                ("Show robot trails", self.show_trails_var)):
            tk.Checkbutton(options, text=label, variable=variable, command=self.render).pack(side="left", padx=3)
        tk.Label(options, text="Arrival tolerance (mm)").pack(side="left", padx=(8, 2))
        self.arrival_tolerance_entry = tk.Entry(options, textvariable=self.arrival_tolerance_var, width=4)
        self.arrival_tolerance_entry.pack(side="left")
        tk.Label(options, text="In-range hold (s)").pack(side="left", padx=(8, 2))
        self.arrival_hold_entry = tk.Entry(options, textvariable=self.arrival_hold_var, width=4)
        self.arrival_hold_entry.pack(side="left")
        self.tracking_status = tk.Label(parent, text="Set Phase 3 robots, then Place ROI and click the camera.", anchor="w", wraplength=1000)
        self.tracking_status.pack(fill="x", padx=10)
        learned_bar = tk.Frame(parent)
        learned_bar.pack(fill='x', padx=8, pady=2)
        transmission = tk.LabelFrame(learned_bar, text='Motor commands')
        transmission.pack(side='right', padx=4)
        self.build_motor_repeat_controls(transmission)
        self.manual_check = tk.Checkbutton(learned_bar, text='Manual mode', variable=self.manual_mode_var,
                                           command=self.change_manual_mode)
        self.manual_check.pack(side='left')
        HoverHint(self.manual_check, 'Select a point in the table, then click the camera to set commanded XY. '
                  'While stopped, Run selected point sends it. During a run, clicks send a new command for the active point. '
                  'Save u keeps the last sent value. '
                  'Automatic control is disabled in Manual mode.')
        self.learned_controls = []
        for label, action in [('Save u', self.save_learned), ('Export points + u', self.export_learned),
                              ('Load u', self.load_learned), ('Clear u', self.clear_learned)]:
            button = tk.Button(learned_bar, text=label, command=action)
            button.pack(side='left', padx=2)
            if label == 'Save u':
                self.save_u_button = button
                HoverHint(button, 'Keep the active point\'s latest sent u in memory and the table. '
                          'When stopped, uses the selected point. No file dialog. Export points + u saves the whole path to disk.')
            elif label == 'Export points + u':
                self.export_u_button = button
            else:
                self.learned_controls.append(button)
        tk.Label(learned_bar, text='XY and u: mm').pack(side='left', padx=6)
        learned_check = tk.Checkbutton(learned_bar, text='Use learned u', variable=self.use_learned_var,
                                     command=self.refresh_point_corrections)
        learned_check.pack(side='left')
        self.learned_controls.append(learned_check)
        HoverHint(learned_check, 'Start each point at desired + its saved manual XY u. '
                  'Force control changes magnet angles separately, not XY u. Memory matches sequence, calibration, hardware and robot count. '
                  'Export points + u for later sessions.')
        force_bar = tk.LabelFrame(parent, text='Force control — one robot')
        force_bar.pack(fill='x', padx=8, pady=3)
        self.force_entries = []
        tk.Label(force_bar, text='Hold speed (mm/s)').pack(side='left', padx=(6, 2))
        hold_speed = tk.Entry(force_bar, textvariable=self.force_hold_speed_var, width=4)
        hold_speed.pack(side='left')
        self.force_entries.append(hold_speed)
        HoverHint(hold_speed, 'Hold the motor pose only when inside arrival tolerance AND below this speed. '
                  'Above this speed, velocity damping continues. This does not estimate disturbances or delay corrections.')
        tk.Label(force_bar, text='Max angle step (deg)').pack(side='left', padx=(6, 2))
        force_step = tk.Entry(force_bar, textvariable=self.force_step_var, width=4)
        force_step.pack(side='left')
        preview = tk.Checkbutton(force_bar, text='Preview only', variable=self.force_preview_var)
        preview.pack(side='left')
        self.force_entries.extend([force_step, preview])
        HoverHint(force_step, 'Maximum model-angle change per PID update; default 10 degrees, also angle-rate limited. '
                  'The solver preserves baseline stiffness and holds on an invalid or infeasible solve.')
        HoverHint(preview, 'Calculate and display force corrections without sending them. '
                  'Initial path and manual commands still move normally.')
        pid_bar = tk.LabelFrame(parent, text='Force PID — gains in µN, mm and seconds (Ki=0 disables integral)')
        pid_bar.pack(fill='x', padx=8, pady=2)
        for key, variable in self.force_pid_settings.items():
            tk.Label(pid_bar, text=key).pack(side='left', padx=(6, 2))
            entry = tk.Entry(pid_bar, textvariable=variable, width=5)
            entry.pack(side='left')
            self.force_entries.append(entry)
            HoverHint(entry, 'Kp: µN/mm; Ki: µN/(mm·s); Kv: µN·s/mm, damping from filtered measured velocity. '
                      'These are starting gains, not hardware-tuned. Period is a maximum update cadence, not guaranteed timing. '
                      'Rate bounds angle changes per elapsed second; deadband suppresses motor-command chatter. '
                      'Integral is bounded and freezes when requested force is not achievable.')
        self.control_details = self.build_control_details(parent)
        recording_bar = tk.Frame(parent)
        recording_bar.pack(fill='x', padx=8, pady=3)
        tk.Button(recording_bar, text='Start recording', command=self.start_recording).pack(side='left', padx=3)
        self.recording_stop_button = tk.Button(recording_bar, text='Stop / save recording', command=self.stop_recording, state='disabled')
        self.recording_stop_button.pack(side='left', padx=3)
        tk.Label(recording_bar, text='Video FPS').pack(side='left', padx=3)
        tk.Entry(recording_bar, textvariable=self.recording_fps_var, width=4).pack(side='left')
        self.recording_status = tk.Label(recording_bar, text='Camera + annotated + blue-screen overlays + tracking CSV', anchor='w')
        self.recording_status.pack(side='left', padx=8)
        crop_bar = tk.Frame(parent)
        crop_bar.pack(fill='x', padx=8, pady=2)
        self.crop_controls = []
        for label, variable in [('Circle crop', self.crop_enabled_var), ('Fit workspace (4 pins)', self.crop_fit_var)]:
            check = tk.Checkbutton(crop_bar, text=label, variable=variable, command=self.render)
            check.pack(side='left')
            self.crop_controls.append(check)
        tk.Label(crop_bar, text='Radius px').pack(side='left')
        crop_radius = tk.Entry(crop_bar, textvariable=self.crop_radius_var, width=5)
        crop_radius.pack(side='left')
        crop_radius.bind('<Return>', lambda event: self.render())
        crop_place = tk.Button(crop_bar, text='Place crop', command=self.begin_crop_placement)
        crop_place.pack(side='left', padx=4)
        self.crop_controls.extend([crop_radius, crop_place])
        HoverHint(crop_place, 'Unchecks Fit workspace. Click the main/pop-out camera to position the export circle. '
                  'Crop is frozen during recording; tracking and CSV stay in native camera coordinates.')
        tk.Label(parent, text="Green: tracked robots    Cyan: desired points    Orange: active target    Red: beyond pins. Select a row or click a target.",
                 anchor="w").pack(fill="x", padx=10)
        self.canvas = tk.Canvas(parent, background="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=(3, 8))
        self.canvas.bind("<Configure>", lambda event: self.render())
        self.canvas.bind("<Button-1>", lambda event: self.camera_click(event, self.canvas))

    def load_sequence(self):
        if self.running or self.precalculating:
            return
        path = show_view_dialog(self, filedialog.askopenfilename, title="Load desired XY sequence", initialdir=str(XY_SEQUENCE_DIR),
                                          filetypes=[("XY sequences", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.load_path(path)
        except (ValueError, OSError) as error:
            show_view_dialog(self, messagebox.showerror, "XY sequence", str(error))
            return

    def change_units(self, event=None):
        if self.running or self.precalculating or self.sequence_path is None:
            return
        try:
            self.load_path(self.sequence_path)
        except (ValueError, OSError) as error:
            self.steps, self.pixels, self.setup = [], None, None
            self.status.config(text=f"Not ready: {error}")
            self.render()
            show_view_dialog(self, messagebox.showerror, "XY sequence units", str(error))

    def load_path(self, path):
        units = self.units_var.get()
        steps = read_xy_sequence(path, units=units)
        self.input_units = resolve_xy_units(Path(path).read_text(encoding="utf-8-sig"), units)
        self.sequence_path = path
        self.steps, self.pixels, self.setup = steps, None, None
        self.active_index = None
        self.file_label.config(text=f"{path} — {len(steps)} points — input units: {self.input_units}; table: mm")
        self.table.delete(*self.table.get_children())
        for index, step in enumerate(steps):
            self.table.insert("", "end", iid=str(index), text=str(index + 1), values=(
                f"{step.target_xy_m[0] * 1000:g}", f"{step.target_xy_m[1] * 1000:g}",
                '0.00', '0.00', f"{step.target_xy_m[0] * 1000:.2f}", f"{step.target_xy_m[1] * 1000:.2f}",
                f"{step.wait_seconds:g}", f"{step.zero_angle_deg:g}", f"{step.ratio:g}",
                "Auto" if step.angle_deg is None else f"{step.angle_deg:g}", "Pending", ""))
        self.refresh_calibration()

    def edit_point_cell(self, event):
        if self.running or self.precalculating:
            return
        row = self.table.identify_row(event.y)
        column = self.table.identify_column(event.x)
        fields = {"#7": "wait", "#8": "zero", "#9": "ratio", "#10": "angle"}
        if not row or column not in fields:
            return
        if self.cell_editor is not None:
            self.cell_editor.destroy()
        key = fields[column]
        x, y, width, height = self.table.bbox(row, column)
        editor = tk.Entry(self.table)
        self.cell_editor = editor
        editor.insert(0, self.table.set(row, key))
        editor.place(x=x, y=y, width=width, height=height)
        editor.focus_set()
        editor.select_range(0, 'end')

        def finish(event=None, cancel=False):
            if self.cell_editor is not editor:
                return
            if not cancel:
                try:
                    text = editor.get().strip()
                    value = None if key == 'angle' and text.lower() in ('auto', 'none') else float(text)
                    if value is not None and not np.isfinite(value):
                        raise ValueError('Value must be finite.')
                    if key == 'ratio' and value <= 0:
                        raise ValueError('Ratio must be positive.')
                    if key == 'wait' and value < 0:
                        raise ValueError('Wait must be nonnegative.')
                    if key == 'zero' and not 0 <= value <= 360:
                        raise ValueError('Motor zero must be between 0 and 360 degrees.')
                    field = {'ratio': 'ratio', 'angle': 'angle_deg', 'wait': 'wait_seconds', 'zero': 'zero_angle_deg'}[key]
                    self.steps[int(row)] = replace(self.steps[int(row)], **{field: value})
                    self.table.set(row, key, 'Auto' if value is None else f'{value:g}')
                    self.table.set(row, 'angles', '')
                    self.table.set(row, 'state', 'Pending')
                    self.refresh_point_corrections()
                    self.status.config(text=f'Point {int(row) + 1}: {key} updated. Export XY sequence to save edits.')
                except ValueError as error:
                    self.status.config(text=f'Point edit rejected: {error}')
            self.cell_editor = None
            editor.destroy()

        editor.bind('<Return>', finish)
        editor.bind('<FocusOut>', finish)
        editor.bind('<Escape>', lambda event: finish(cancel=True))

    def export_sequence(self):
        if not self.steps:
            show_view_dialog(self, messagebox.showerror, 'Sequence export', 'Load a sequence first.')
            return
        path = show_view_dialog(self, filedialog.asksaveasfilename, title='Save edited XY sequence', initialdir=str(XY_SEQUENCE_DIR),
                                           initialfile=Path(self.sequence_path).name, defaultextension='.txt',
                                           filetypes=[('XY sequence', '*.txt')])
        if path:
            try:
                write_xy_sequence(path, self.steps)
                self.status.config(text=f'Saved XY, wait, motor zero, ratio and equilibrium angle: {path}')
            except OSError as error:
                show_view_dialog(self, messagebox.showerror, 'Sequence export', str(error))

    def apply_point_settings(self):
        if self.running or getattr(self, 'precalculating', False):
            return
        selection = self.table.selection()
        if not selection:
            self.status.config(text='Select a point before applying ratio and angle.')
            return
        try:
            ratio = float(self.point_ratio_var.get())
            text = self.point_angle_var.get().strip()
            angle = None if text.lower() in ('auto', 'none') else float(text)
            if not np.isfinite(ratio) or ratio <= 0 or (angle is not None and not np.isfinite(angle)):
                raise ValueError('Ratio must be positive and finite; angle must be finite or Auto.')
            index = int(selection[0])
            self.steps[index] = replace(self.steps[index], ratio=ratio, angle_deg=angle)
            self.table.set(str(index), 'ratio', f'{ratio:g}')
            self.table.set(str(index), 'angle', 'Auto' if angle is None else f'{angle:g}')
            self.table.set(str(index), 'angles', '')
            self.table.set(str(index), 'state', 'Pending')
            self.refresh_point_corrections()
            self.status.config(text=f'Point {index + 1} settings applied. Export XY sequence to save them.')
        except ValueError as error:
            self.status.config(text=f'Point edit rejected: {error}')

    def current_setup(self):
        setup = json.loads(json.dumps(self.calibration.setup_data(require_capture=False)))
        if not setup.get("control_case"):
            raise ValueError("Select the hardware case in Calibration first.")
        if self.frame is not None and list(self.frame.shape[1::-1]) != setup["video_size_px"]:
            raise ValueError("Live camera size differs from calibration. Recalibrate or load a matching setup.")
        return setup

    def prepare(self, *, validate=True):
        if not self.steps:
            raise ValueError("Load an XY sequence first.")
        setup = self.current_setup()
        pixels = project_sequence(self.steps, setup, validate=validate)
        return setup, pixels

    def validate_corrected_target(self, target, setup, *, allow_outside_pins):
        target = np.asarray(target, dtype=float)
        if target.shape != (2,) or not np.isfinite(target).all():
            raise ValueError('Corrected XY must be a finite X,Y pair.')
        corrected = [replace(self.steps[0], target_xy_m=tuple(target))]
        pixels = project_sequence(corrected, setup, validate=False)
        validate_projected_points(corrected, setup, pixels, allow_outside_pins=allow_outside_pins)

    def refresh_calibration(self):
        if self.running:
            return
        if hasattr(self, 'next_button'):
            self.refresh_next_button()
        try:
            setup = self.current_setup()
            self.configure_tracking(setup)
            self.refresh_point_corrections(setup)
            if not self.steps:
                self.setup, self.pixels = setup, None
                self.status.config(text='Calibration ROI/settings applied. Check Phase 3 robots; reposition ROI if needed, then load a sequence.')
                self.render()
                return
            self.setup, self.pixels = self.prepare(validate=False)
            outside, off_frame = point_limits(self.pixels, self.setup)
            self.limits_label.config(text=sequence_diagnostics(self.steps, self.setup, self.pixels))
            for index in range(len(self.steps)):
                self.table.item(str(index), tags=("outside",) if outside[index] else ())
                if self.table.set(str(index), "state") in ("Pending", "Off camera", "Beyond pins"):
                    self.table.set(str(index), "state", "Off camera" if off_frame[index] else "Beyond pins" if outside[index] else "Pending")
            self.status.config(text=f"Preview: {len(self.steps)} points; input {self.input_units}, table mm; rectangle {self.setup['rectangle_width_cm']:g} × {self.setup['rectangle_height_cm']:g} cm. "
                               f"{int(outside.sum())} beyond pins, {int(off_frame.sum())} off camera. Select an inside point to run, or review the extrapolation option.")
        except (ValueError, TypeError, KeyError) as error:
            self.setup, self.pixels = None, None
            self.tracking_tracker = None
            self.tracking_detections = []
            self.tracking_group_xy = None
            self.status.config(text=f"Not ready: {error}")
            self.limits_label.config(text="")
            if getattr(self, 'command_roi', None) is not None:
                self.start_tracking()  # Pixel-only detection remains available.
        self.render()

    def accept_live_frame(self, frame):
        self.frame = frame
        self.frame_received_at = time.monotonic()
        if self.setup is not None and list(frame.shape[1::-1]) != self.setup["video_size_px"]:
            self.stop_event.set()
            self.preparation_stop.set()
            self.setup = None  # Do not draw old pins/targets on a new frame geometry.
            self.tracking_signature = None
            if hasattr(self, 'command_roi'):
                self.command_roi = None
            self.pixels = None
            self.tracking_tracker = None
            self.tracking_detections = []
            self.tracking_group_xy = None
            self.tracking_status.config(text="Tracking stopped: camera size no longer matches calibration.")
            self.status.config(text="Camera resolution changed. Sequence stopped; load matching calibration before running again.")
        first_row = len(self.tracking_rows)
        if getattr(self, "tracking_tracker", None) is not None:
            self.process_tracking_frame()
        if getattr(self, 'diagnostics', None) is not None:
            self.record_diagnostic_frame(self.tracking_rows[first_row:])
        if getattr(self, 'recording', None) is not None:
            try:
                options = dict(show_path=self.show_path_var.get(), show_roi=self.show_roi_var.get(),
                               commanded_pixel=self.commanded_pixel() if self.show_commanded_var.get() else None,
                               trails=self.robot_trails if self.show_trails_var.get() else None)
                args = (self.setup, self.pixels, self.active_index, self.tracking_detections, self.tracking_tracker)
                annotated = draw_command_overlay(frame.copy(), *args, **options)
                overlay = np.zeros_like(frame)
                overlay[:] = (255, 0, 0)  # Blue chroma-key background; MP4 has no alpha channel.
                draw_command_overlay(overlay, *args, **options)
                rows = self.tracking_rows[first_row:]
                if not rows:
                    rows = [{'frame': '', 'group_valid': 0, 'detected_robots': 0}]
                self.recording.write(frame, annotated, overlay, rows)
                self.recording_status.config(text=f'Recording {self.recording.frame_count} frames; {self.recording.output_size[0]} × {self.recording.output_size[1]} videos + native-coordinate CSV.')
            except (ValueError, OSError, cv2.error) as error:
                self.stop_recording()
                self.recording_status.config(text=f'Recording stopped: {error}. Partial files kept.')
        if self.visible():
            self.render()

    def configure_tracking(self, setup, *, reset=False):
        if setup.get("coordinate_system") != "robot_axes":
            raise ValueError("Capture Center, +X and +Y, or load a complete robot-axis calibration first.")
        signature = json.dumps({key: setup.get(key) for key in (
            "corners_px", "rectangle_width_cm", "rectangle_height_cm", "axis_samples_px", "video_size_px",
            "roi_circle_px", "minimum_area", "dark_contrast_threshold")}, sort_keys=True)
        revision = getattr(self.calibration, 'setup_revision', 0)
        revision = revision if isinstance(revision, int) else 0
        loaded_changed = getattr(self, 'calibration_revision', None) != revision
        if loaded_changed or reset:
            self.roi_override = False
        self.calibration_revision = revision
        if not reset and not loaded_changed and signature == self.tracking_signature:
            if self.tracking_enabled and self.tracking_tracker is None and (not hasattr(self, 'command_roi') or self.command_roi is not None):
                self.start_tracking(setup=setup)
            return
        self.tracking_signature = signature
        self.setup = setup
        if hasattr(self, 'command_roi') and not self.roi_override:
            self.command_roi = tuple(setup['roi_circle_px'])
            self.tracking_enabled = True
            self.show_roi_var.set(True)
        self.tracking_area_var.set(str(setup["minimum_area"]))
        self.tracking_contrast_var.set(str(setup.get("dark_contrast_threshold", 10)))
        if not hasattr(self, 'command_roi') or not getattr(self, 'roi_override', False):
            self.tracking_radius_var.set(str(setup["roi_circle_px"][2]))
        self.tracking_tracker = None
        self.tracking_detections = []
        self.tracking_group_xy = None
        if self.tracking_enabled and (not hasattr(self, 'command_roi') or self.command_roi is not None):
            self.start_tracking(setup=setup)

    def new_tracking_engine(self, setup, center=None):
        count = int(self.tracking_count_var.get())
        area, contrast = float(self.tracking_area_var.get()), float(self.tracking_contrast_var.get())
        radius = float(self.tracking_radius_var.get())
        circle = getattr(self, 'command_roi', None) or setup['roi_circle_px']
        center = circle[:2] if center is None else center
        return MovingROICalibration(
            setup["corners_px"], setup["rectangle_width_cm"], setup["rectangle_height_cm"],
            (*center, radius), count, area, setup.get('axis_samples_px') or None,
            contrast_threshold=contrast, limit_to_pins=False, allow_uncalibrated=True,
        )

    def start_tracking(self, *, setup=None):
        if getattr(self, "arrival_gate", None) is not None:
            self.arrival_gate.observe(None)
            self.refresh_next_button()
        try:
            if hasattr(self, 'command_roi') and self.command_roi is None:
                raise ValueError('Place the Phase 3 ROI around the robots first.')
            if setup is None:
                try:
                    setup = self.current_setup()
                except ValueError:
                    if self.frame is None:
                        raise ValueError('Connect the camera and wait for a frame first.')
                    setup = {'coordinate_system':'pixels', 'corners_px':[], 'rectangle_width_cm':10,
                             'rectangle_height_cm':10, 'video_size_px':list(self.frame.shape[1::-1]),
                             'axis_samples_px':{}, 'roi_circle_px':self.command_roi}
            tracker = self.new_tracking_engine(setup)
        except (ValueError, TypeError, KeyError) as error:
            self.tracking_status.config(text=f"Cannot start tracking: {error}")
            return
        self.setup = setup
        self.tracking_tracker, self.tracking_enabled = tracker, True
        self.tracking_started_at = time.monotonic()
        self.tracking_frame_number = 0
        self.tracking_rows, self.robot_trails, self.tracking_detections = [], {}, []
        self.tracking_group_xy = None
        if (self.frame is not None and self.frame_received_at is not None
                and time.monotonic() - self.frame_received_at <= 2):
            self.process_tracking_frame()
        else:
            self.tracking_status.config(text="Tracking ready; waiting for a live camera frame.")
        self.render()

    def stop_tracking(self):
        self.tracking_enabled = False
        self.tracking_tracker = None
        self.tracking_detections = []
        self.tracking_group_xy = None
        if getattr(self, "arrival_gate", None) is not None:
            self.arrival_gate.observe(None)
            self.refresh_next_button()
        self.tracking_status.config(text="Tracking stopped. Recorded measurements remain available for export; motor execution is unchanged.")
        self.render()

    def process_tracking_frame(self):
        try:
            tracker = self.tracking_tracker
            if tracker.count != int(self.tracking_count_var.get()):
                if self.running:
                    raise ValueError('Robot count cannot change during a run. Stop first or restore the previous count.')
                tracker = self.new_tracking_engine(self.setup, center=tracker.center)
                self.tracking_tracker = tracker
                self.robot_trails = {}
            tracker.minimum_area = float(self.tracking_area_var.get())
            tracker.contrast_threshold = float(self.tracking_contrast_var.get())
            tracker.radius = float(self.tracking_radius_var.get())
            if (not np.isfinite([tracker.minimum_area, tracker.contrast_threshold, tracker.radius]).all()
                    or tracker.minimum_area <= 0 or tracker.radius <= 0 or not 1 <= tracker.contrast_threshold <= 255):
                raise ValueError("Min area and ROI radius must be positive; contrast must be 1–255.")
            self.tracking_detections, center, self.tracking_group_xy = tracker.process(self.frame)
            if getattr(self, 'command_roi', None) is not None and center is not None:
                self.command_roi = (*tracker.center, tracker.radius)
        except (ValueError, TypeError) as error:
            self.tracking_detections, self.tracking_group_xy = [], None
            if getattr(self, "arrival_gate", None) is not None:
                self.arrival_gate.observe(None)
                self.refresh_next_button()
            self.tracking_status.config(text=f"Tracking unavailable: {error}")
            return
        if getattr(self, "arrival_gate", None) is not None:
            positions = (None if center is None else
                         np.asarray([item["xy_cm"] for item in sorted(self.tracking_detections, key=lambda item: item['robot_id'])]) * .01)
            self.arrival_gate.observe(positions, self.frame_received_at)
            self.refresh_next_button()
        target = None if self.active_index is None else self.steps[self.active_index].target_xy_m
        group_mm = None if self.tracking_group_xy is None else np.asarray(self.tracking_group_xy) * 10
        error_mm = None if target is None or group_mm is None else float(np.linalg.norm(group_mm - np.asarray(target) * 1000))
        visible_ids = {item["robot_id"] for item in self.tracking_detections}
        for robot_id, trail in self.robot_trails.items():
            if robot_id not in visible_ids:
                trail.append(None)
        for item in self.tracking_detections:
            self.robot_trails.setdefault(item["robot_id"], deque(maxlen=300)).append(item["center"])
        elapsed = time.monotonic() - self.tracking_started_at
        gate = getattr(self, "arrival_gate", None)
        feedback = getattr(self, 'feedback', None)
        control_data = {} if feedback is None else feedback.snapshot()
        for item in self.tracking_detections or [None]:
            self.tracking_rows.append({
                "frame": self.tracking_frame_number, "time_s": elapsed,
                "robot_id": "" if item is None else item["robot_id"],
                "robot_x_px": "" if item is None else item["center"][0],
                "robot_y_px": "" if item is None else item["center"][1],
                "robot_x_mm": "" if item is None or item['xy_cm'] is None else item["xy_cm"][0] * 10,
                "robot_y_mm": "" if item is None or item['xy_cm'] is None else item["xy_cm"][1] * 10,
                "group_x_mm": "" if group_mm is None else group_mm[0],
                "group_y_mm": "" if group_mm is None else group_mm[1],
                "expected_robots": tracker.count, "detected_robots": len(self.tracking_detections),
                "group_valid": int(center is not None),
                "active_point": "" if self.active_index is None else self.active_index + 1,
                "desired_x_mm": "" if target is None else target[0] * 1000,
                "desired_y_mm": "" if target is None else target[1] * 1000,
                "error_mm": "" if error_mm is None else error_mm,
                "roi_x_px": tracker.center[0], "roi_y_px": tracker.center[1], "roi_radius_px": tracker.radius,
                "sequence_running": int(self.running),
                "arrival_ready": int(gate.ready) if gate is not None else 0,
                "arrival_max_error_mm": "" if gate is None or gate.error_mm is None else gate.error_mm,
                "desired_ratio": "" if self.active_index is None else self.steps[self.active_index].ratio,
                "desired_angle_deg": "" if self.active_index is None or self.steps[self.active_index].angle_deg is None else self.steps[self.active_index].angle_deg,
                'control_enabled': int(feedback.enabled.is_set()) if feedback is not None else 0,
                'control_state': control_data.get('state', 'Off'),
                'control_mode': control_data.get('mode', ''),
                'robot_speed_mm_s': control_data.get('robot_speed_mm_s', ''),
                'requested_force_n': json.dumps(control_data.get('requested_force_n')),
                'predicted_force_n': json.dumps(control_data.get('predicted_force_n')),
                'force_residual_n': control_data.get('force_residual_n', ''),
                'magnet_cosine_inputs': json.dumps(control_data.get('magnet_inputs')),
                'stiffness_eigenvalues_n_m': json.dumps(control_data.get('stiffness_eigenvalues')),
                'control_calculation_time_s': control_data.get('calculation_time_s', ''),
                'force_compensated': int(control_data.get('force_compensated', False)),
                'velocity_xy_mm_s': json.dumps(control_data.get('velocity_xy_mm_s')),
                'position_error_mm': json.dumps(control_data.get('position_error_mm')),
                'pid_force_n': json.dumps(control_data.get('pid_force_n')),
                'integral_force_n': json.dumps(control_data.get('integral_force_n')),
                'hold_speed_mm_s': control_data.get('hold_speed_mm_s', ''),
                'observation_age_s': control_data.get('observation_age_s', ''),
                'control_cycle_time_s': control_data.get('control_cycle_time_s', ''),
                'force_kp_n_m': control_data.get('force_kp_n_m', ''),
                'force_ki_n_m_s': control_data.get('force_ki_n_m_s', ''),
                'force_kv_n_s_m': control_data.get('force_kv_n_s_m', ''),
                'correction_x_mm': (control_data.get('correction_mm') or ['', ''])[0],
                'correction_y_mm': (control_data.get('correction_mm') or ['', ''])[1],
                'commanded_x_mm': (control_data.get('commanded_xy_mm') or ['', ''])[0],
                'commanded_y_mm': (control_data.get('commanded_xy_mm') or ['', ''])[1],
                'proposed_xy_mm': json.dumps(control_data.get('proposed_xy_mm')),
                'calculated_motor_angles_deg': json.dumps(control_data.get('calculated_angles', [])),
                'sent_motor_angles_deg': json.dumps(control_data.get('sent_angles', [])),
                'motor_sent_time_s': control_data.get('sent_time_s', ''),
                'orientation_measured_deg': '',
            })
        self.tracking_frame_number += 1
        detail = "Group missing; ROI held at last complete detection."
        if center is not None and group_mm is None:
            detail = f'Robots detected; pixel center {center}. Physical XY requires complete calibration.'
        if group_mm is not None:
            detail = f"Group XY ({group_mm[0]:.2f}, {group_mm[1]:.2f}) mm."
        if error_mm is not None:
            detail += f" Active-target error {error_mm:.2f} mm."
        if gate is not None and gate.error_mm is not None:
            detail += f" Arrival maximum robot error {gate.error_mm:.2f} mm (limit {gate.tolerance_mm:g} mm)."
        self.tracking_status.config(text=f"Tracking {len(self.tracking_detections)}/{tracker.count} robots. {detail}")
        if hasattr(self, 'control_details'):
            self.refresh_control_details(self.control_details)

    def check_camera_health(self):
        if self.tracking_tracker is None:
            return
        if not self.camera_running() or self.frame_received_at is None or time.monotonic() - self.frame_received_at > 2:
            had_detections = bool(self.tracking_detections)
            self.tracking_detections, self.tracking_group_xy = [], None
            if getattr(self, "arrival_gate", None) is not None:
                self.arrival_gate.observe(None)
                self.refresh_next_button()
            if had_detections:
                for trail in self.robot_trails.values():
                    trail.append(None)
            self.tracking_status.config(text="No fresh camera stream. Robot positions are unavailable; ROI held until video resumes.")
            if had_detections and self.visible():
                self.render()

    def export_tracking(self):
        if not self.tracking_rows:
            show_view_dialog(self, messagebox.showerror, "Tracking export", "No measurements recorded. Start tracking with a live camera first.")
            return
        path = show_view_dialog(self, filedialog.asksaveasfilename, title="Save robot tracking", initialdir=str(XY_SEQUENCE_DIR.parent),
                                           initialfile=default_setup_name().replace(".json", "_tracking.csv"),
                                           defaultextension=".csv", filetypes=[("Tracking measurements", "*.csv")])
        if path:
            try:
                with open(path, "w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(self.tracking_rows[0]))
                    writer.writeheader()
                    writer.writerows(self.tracking_rows)
                self.tracking_status.config(text=f"Exported {len(self.tracking_rows)} tracking measurements.")
            except OSError as error:
                show_view_dialog(self, messagebox.showerror, "Tracking export", str(error))

    def diagnostic_settings(self):
        names = ('tracking_count_var', 'tracking_radius_var', 'tracking_area_var',
                 'tracking_contrast_var', 'arrival_tolerance_var', 'arrival_hold_var',
                 'force_step_var', 'force_preview_var', 'force_hold_speed_var',
                 'manual_mode_var', 'repeat_motors_var', 'motor_repeat_count_var',
                 'allow_extrapolation_var', 'use_learned_var')
        values = {name: getattr(self, name).get() for name in names if hasattr(self, name)}
        values['pid_display_units'] = {key: var.get() for key, var in self.force_pid_settings.items()}
        values['controller'] = 'force_pid_without_disturbance_learning'
        values['camera_source'] = getattr(self.calibration, 'camera_source_key', None)
        values['processing_fps'] = self.calibration.live_fps_var.get()
        return values

    def diagnostic_metadata(self):
        setup = self.setup
        if setup is None:
            try:
                setup = self.calibration.setup_data(require_capture=False)
            except (ValueError, TypeError, AttributeError):
                setup = None
        metadata = dict(settings=self.diagnostic_settings(), calibration=setup,
                        sequence_path=self.sequence_path, input_units=self.input_units,
                        steps=self.steps, displayed_corrections=self.display_corrections,
                        software=dict(numpy=np.__version__, opencv=cv2.__version__))
        if setup and setup.get('control_case'):
            try:
                metadata['hardware_model'] = load_hardware_case(setup['control_case'])
            except Exception as error:
                metadata['hardware_model_error'] = str(error)
        return metadata

    def toggle_diagnostics(self):
        if self.diagnostics is not None:
            self.stop_diagnostics()
            return
        try:
            recorder = ControlDiagnostics(Path(__file__).resolve().parent / 'diagnostics',
                                          self.diagnostic_metadata())
        except (OSError, ValueError, TypeError) as error:
            self.status.config(text=f'Cannot start diagnostics: {error}')
            return
        self.diagnostics = recorder
        self.diagnostic_frame = 0
        self.diagnostic_previous_receipt = None
        self.commander.diagnostic_sink = recorder.record
        self.diagnostics_button.config(text='Stop diagnostics')
        self.status.config(text=f'Diagnostics recording: {recorder.path}. Stop diagnostics to save; no video included.')
        self.poll_diagnostics()

    def poll_diagnostics(self):
        self.diagnostics_after = None
        recorder = self.diagnostics
        if recorder is None:
            return
        # Heartbeats show stream outages and control-state changes even with no frames.
        recorder.record('heartbeat', dict(settings=self.diagnostic_settings(), running=self.running,
            camera_running=self.camera_running(), frame_age_s=None if self.frame_received_at is None else time.monotonic()-self.frame_received_at,
            active_index=self.active_index, status=self.status.cget('text'),
            tracking_status=self.tracking_status.cget('text'),
            control={} if self.feedback is None else self.feedback.snapshot()))
        if recorder.error:
            self.status.config(text=f'{recorder.error} Partial diagnostics: {recorder.path}')
            self.diagnostics_button.config(text='Stop diagnostics (!)')
        self.diagnostics_after = self.parent.after(500, self.poll_diagnostics)

    def record_diagnostic_frame(self, rows):
        recorder = self.diagnostics
        now = time.monotonic()
        index = self.active_index
        active_step = self.steps[index] if index is not None and 0 <= index < len(self.steps) else None
        recorder.record('frame', dict(diagnostic_frame=self.diagnostic_frame,
            receipt_monotonic_s=self.frame_received_at, elapsed_s=now-recorder.started_monotonic,
            inter_frame_s=None if self.diagnostic_previous_receipt is None else self.frame_received_at-self.diagnostic_previous_receipt,
            width_px=self.frame.shape[1], height_px=self.frame.shape[0],
            processing_s=now-self.frame_received_at, tracking_active=self.tracking_tracker is not None,
            tracking_status=self.tracking_status.cget('text'), status=self.status.cget('text'),
            settings=self.diagnostic_settings(), calibration=self.setup,
            active_index=index, active_step=active_step, running=self.running,
            active_target_px=None if index is None or self.pixels is None else self.pixels[index],
            control_enabled=self.feedback is not None and self.feedback.enabled.is_set(),
            robot_selection_error=getattr(self.tracking_tracker, 'selection_error', None),
            command_roi=self.command_roi, detections=self.tracking_detections,
            group_xy_cm=self.tracking_group_xy, tracking_rows=rows,
            control={} if self.feedback is None else self.feedback.snapshot()))
        self.diagnostic_previous_receipt = self.frame_received_at
        self.diagnostic_frame += 1

    def stop_diagnostics(self):
        recorder, self.diagnostics = self.diagnostics, None
        if recorder is None:
            return
        self.commander.diagnostic_sink = None
        if getattr(self, 'diagnostics_after', None) is not None:
            self.parent.after_cancel(self.diagnostics_after)
            self.diagnostics_after = None
        try:
            final_metadata = self.diagnostic_metadata()
        except Exception as error:
            # Always drain/close the writer, even with incomplete GUI settings.
            final_metadata = dict(metadata_error=str(error))
        try:
            recorder.close(final_metadata)
            message = f'Saved diagnostics: {recorder.path} ({recorder.frames} frames).'
            if recorder.error:
                message += f' INCOMPLETE: {recorder.error}'
        except (OSError, ValueError, TypeError) as error:
            message = f'Diagnostic finalization failed: {error}. Partial files: {recorder.path}'
        self.diagnostics_button.config(text='Start diagnostics')
        self.status.config(text=message)

    def start_recording(self):
        if self.recording is not None:
            return
        if not self.camera_running() or self.frame_received_at is None or time.monotonic() - self.frame_received_at > 2:
            show_view_dialog(self, messagebox.showerror, 'Recording', 'Connect the camera and wait for a fresh frame first.')
            return
        if self.tracking_tracker is None or not self.tracking_enabled:
            show_view_dialog(self, messagebox.showerror, 'Recording', 'Apply calibration and start tracking first to record synchronized measurements.')
            return
        path = show_view_dialog(self, filedialog.asksaveasfilename, title='Choose recording base name (four files will be created)',
                                           initialdir=str(XY_SEQUENCE_DIR.parent),
                                           initialfile=default_setup_name().replace('.json', '.mp4'),
                                           defaultextension='.mp4', filetypes=[('Recording', '*.mp4')])
        if not path:
            return
        try:
            circle = self.recording_circle()
            self.recording = CommandRecording(path, self.frame.shape[1::-1], float(self.recording_fps_var.get()), crop_circle=circle)
            self.crop_placing = False
            for widget in self.crop_controls:
                widget.config(state='disabled')
            self.recording_stop_button.config(state='normal')
            self.recording_status.config(text='Recording started. Stop / save finalizes all four files.')
        except (ValueError, OSError, cv2.error) as error:
            show_view_dialog(self, messagebox.showerror, 'Recording', str(error))

    def stop_recording(self):
        if self.recording is not None:
            recording, self.recording = self.recording, None
            recording.close()
            for widget in getattr(self, 'crop_controls', []):
                widget.config(state='normal')
            self.recording_stop_button.config(state='disabled')
            self.recording_status.config(text=f'Saved {recording.frame_count} frames: {recording.paths[0].parent / recording.paths[0].stem} (+ annotated, overlay, CSV).')

    def select_point(self, event=None):
        if not self.running and self.table.selection():
            self.active_index = int(self.table.selection()[0])
            if hasattr(self, 'point_ratio_var'):
                step = self.steps[self.active_index]
                self.point_ratio_var.set(f'{step.ratio:g}')
                self.point_angle_var.set('Auto' if step.angle_deg is None else f'{step.angle_deg:g}')
            self.render()
            self.refresh_next_button()

    def begin_roi_placement(self):
        if self.running or self.precalculating:
            self.tracking_status.config(text='Stop the run/precalculation before changing the ROI.')
            return
        self.roi_placement_var.set(True)
        self.place_roi_button.config(text='Click camera')
        self.tracking_status.config(text='Click the ROI center around all Phase 3 robots; detection starts automatically.')

    def camera_click(self, event, canvas):
        if getattr(self, 'crop_placing', False):
            transform = getattr(canvas, 'camera_transform', None)
            if transform is not None and self.frame is not None and self.recording is None:
                ox, oy, scale = transform
                point = ((event.x-ox)/scale, (event.y-oy)/scale)
                h,w = self.frame.shape[:2]
                if 0 <= point[0] < w and 0 <= point[1] < h:
                    self.crop_center, self.crop_placing = point, False
                    self.render()
            return
        if not self.roi_placement_var.get():
            if getattr(self, 'manual_mode_var', None) is not None and self.manual_mode_var.get():
                self.set_manual_target(event, canvas)
                return
            self.click_target(event, screen_points=getattr(canvas, 'screen_points', None))
            return
        if self.running or self.precalculating:
            return
        try:
            if self.frame is None or self.frame_received_at is None or time.monotonic() - self.frame_received_at > 2:
                raise ValueError('Wait for a fresh camera frame before placing the ROI.')
            transform = getattr(canvas, 'camera_transform', None)
            if transform is None:
                return
            ox, oy, scale = transform
            x, y = (event.x - ox) / scale, (event.y - oy) / scale
            h, w = self.frame.shape[:2]
            if not (0 <= x < w and 0 <= y < h):
                return
            radius = float(self.tracking_radius_var.get())
            if not np.isfinite(radius) or radius <= 0 or int(self.tracking_count_var.get()) < 1:
                raise ValueError('Set a positive ROI radius and Phase 3 robot count.')
            self.command_roi = (x, y, radius)
            self.roi_override = True
            self.show_roi_var.set(True)
            self.roi_placement_var.set(False)
            self.place_roi_button.config(text='Place ROI')
            self.start_tracking()
        except (ValueError, TypeError) as error:
            self.tracking_status.config(text=f'ROI not placed: {error}')

    def clear_command_roi(self):
        if self.running or self.precalculating:
            self.stop()
        if getattr(self, 'feedback', None) is not None:
            self.feedback.enabled.clear()
        self.command_roi = None
        self.roi_override = True
        self.roi_placement_var.set(False)
        self.place_roi_button.config(text='Place ROI')
        self.show_roi_var.set(False)
        self.robot_trails = {}
        # Clear all visible state before stop_tracking redraws either camera view.
        self.stop_tracking()
        self.tracking_status.config(text='ROI cleared. Set robot count and Place ROI to restart detection.')

    def open_camera_popout(self):
        if self.camera_popout is not None and self.camera_popout.is_open:
            self.camera_popout.show()
            return
        self.camera_popout = CameraPopout(
            self.parent, 'GUI005 — Phase 3 camera', self.render,
            status_lines=2,
            toolbar_builder=self.build_popout_roi_controls,
            toolbar_title='Tracking',
            extra_controls_enabled=lambda: not self.running and not self.precalculating,
            actions=(('Place ROI', self.begin_roi_placement, self.place_roi_button),
                     ('Clear ROI', self.clear_command_roi, None),
                     ('First', self.run, self.run_button),
                     ('Selected', self.run_selected, self.run_selected_button),
                     ('Next', self.next_point, self.next_button),
                     ('Stop', self.stop, self.stop_button),
                     ('Control off', self.toggle_control, self.control_button),
                     ('Save u', self.save_learned, self.save_u_button),
                     ('Crop', self.begin_crop_placement, None),
                     ('Start diagnostics', self.toggle_diagnostics, self.diagnostics_button)),
            action_groups={'Place ROI': 'ROI', 'Clear ROI': 'ROI', 'First': 'Path',
                           'Selected': 'Path', 'Next': 'Path', 'Stop': 'Path',
                           'Control off': 'Control', 'Save u': 'Control', 'Crop': 'View',
                           'Start diagnostics': 'Data'},
            group_builders={'Control': self.build_popout_control_mode},
            action_captions={'Run all': 'First', 'Run selected point': 'Selected', 'Next point': 'Next'},
            status_builder=self.build_popout_control_details, status_refresh=self.refresh_popout_control_details,
            render_version=lambda: (self.render_revision, self.frame_received_at),
            on_close=self.render,
            status=lambda: f"{self.status.cget('text')}  {self.tracking_status.cget('text')}",
        )
        canvas = self.camera_popout.canvas
        canvas.bind('<Button-1>', lambda event: self.camera_click(event, canvas))
        self.render()

    def build_popout_roi_controls(self, bar):
        self.popout_roi_entries = {}
        for label, variable, hint in (
                ('Robots', self.tracking_count_var, 'Expected number of robots in Phase 3. A complete group is required for arrival and feedback.'),
                ('ROI px', self.tracking_radius_var, 'Moving circular ROI radius in native camera pixels.'),
                ('Area', self.tracking_area_var, 'Minimum detected robot area in native pixels squared.'),
                ('Contrast', self.tracking_contrast_var, 'Dark contrast threshold (1–255). Lower values detect fainter robots.')):
            tk.Label(bar, text=label).pack(side='left', padx=(4, 1))
            entry = tk.Entry(bar, textvariable=variable, width=4)
            entry.pack(side='left')
            self.popout_roi_entries[label] = entry
            HoverHint(entry, hint)

    def build_popout_control_mode(self, parent):
        self.popout_manual_check = tk.Checkbutton(parent, text='Manual', variable=self.manual_mode_var,
                                                command=self.change_manual_mode)
        self.popout_manual_check.pack(side='left', padx=2)
        HoverHint(self.popout_manual_check, 'Manual: click to set commanded XY, use Selected to send, then Save u to keep it. '
                  'During a run, clicks send updates to the active point. Modes can change while running.')
        self.build_motor_repeat_controls(parent)

    def build_motor_repeat_controls(self, parent):
        repeat = tk.Checkbutton(parent, text='Repeat', variable=self.repeat_motors_var,
                                command=self.update_motor_repeats)
        repeat.pack(side='left', padx=(4, 0))
        count = ttk.Combobox(parent, values=('2', '3'), textvariable=self.motor_repeat_count_var,
                             state='readonly' if self.repeat_motors_var.get() else 'disabled', width=2)
        count.pack(side='left', padx=(0, 3))
        count.bind('<<ComboboxSelected>>', lambda event: self.update_motor_repeats())
        self.motor_repeat_selectors.append(count)
        HoverHint(repeat, 'Checked: send each motor target 2 or 3 times, with 10 ms between copies. '
                  'Unchecked: send once. Shared with Calibration axis moves and all Commands moves. '
                  'Applies to the next transmission; repeats do not recalculate or add to the angle.')

    def update_motor_repeats(self):
        count = int(self.motor_repeat_count_var.get()) if self.repeat_motors_var.get() else 1
        self.commander.set_motor_repetitions(count)
        self.motor_repeat_selectors = [widget for widget in self.motor_repeat_selectors if widget.winfo_exists()]
        for widget in self.motor_repeat_selectors:
            widget.config(state='readonly' if self.repeat_motors_var.get() else 'disabled')

    def build_popout_control_details(self, parent):
        options = tk.Frame(parent)
        options.pack(fill='x', padx=8, pady=2)
        for label, variable in (('Desired lines', self.show_path_var), ('Moving ROI', self.show_roi_var),
                                ('Commanded point', self.show_commanded_var),
                                ('Robot trails', self.show_trails_var)):
            tk.Checkbutton(options, text=label, variable=variable, command=self.render).pack(side='left', padx=3)
        self.popout_arrival_entries = []
        for label, variable, source, hint in (
                ('Tolerance (mm)', self.arrival_tolerance_var, self.arrival_tolerance_entry,
                 'Maximum distance from the desired point for arrival; shared with the main tab.'),
                ('In-range hold (s)', self.arrival_hold_var, self.arrival_hold_entry,
                 'Continuous time inside arrival tolerance for arrival indication; Next remains manual.')):
            tk.Label(options, text=label).pack(side='left', padx=(8,2))
            entry = tk.Entry(options, textvariable=variable, width=4, state=source.cget('state'))
            entry.pack(side='left')
            self.popout_arrival_entries.append((entry, source))
            HoverHint(entry, hint)
        return self.build_control_details(parent)

    def refresh_popout_control_details(self, label):
        self.popout_manual_check.config(state='disabled' if self.precalculating else 'normal')
        for entry, source in self.popout_arrival_entries:
            entry.config(state=source.cget('state'))
        self.refresh_control_details(label)

    def click_target(self, event, *, screen_points=None):
        screen_points = getattr(self, 'screen_points', None) if screen_points is None else screen_points
        if self.running or self.pixels is None or screen_points is None:
            return
        distances = np.linalg.norm(screen_points - [event.x, event.y], axis=1)
        index = int(np.argmin(distances))
        if distances[index] <= 12:
            self.table.selection_set(str(index))
            self.table.see(str(index))
            self.select_point()

    def run_selected(self):
        self.run(selected_only=True)

    def toggle_control(self):
        if getattr(self, 'manual_mode_var', None) is not None and self.manual_mode_var.get():
            self.manual_mode_var.set(False)
            self.change_manual_mode()
        enabled = not self.control_requested
        self.control_requested = enabled
        self.control_button.config(text='Control on' if enabled else 'Control off')
        if self.feedback is not None:
            self.feedback.set_enabled(enabled)

    def build_control_details(self, parent):
        panel = tk.LabelFrame(parent, text='Control details (XY mm; model force µN)')
        panel.pack(fill='x', padx=8, pady=3)
        fields = {}
        rows = (
            (('desired', 'Desired'), ('measured', 'Measured'), ('u', 'u'), ('commanded', 'Commanded')),
            (('ratio', 'Ratio'), ('angle', 'Angle'), ('speed', 'Speed mm/s'),
             ('error', 'Error mm'), ('state', 'State')),
            (('integral', 'Integral µN'), ('feedback_force', 'Requested µN'),
             ('allocated_force', 'Allocated µN'), ('residual', 'Residual µN')),
            (('calculated', 'Calculated motors'), ('sent', 'Last sent motors')),
        )
        for row in rows:
            strip = tk.Frame(panel)
            strip.pack(fill='x', padx=6)
            for column, (key, caption) in enumerate(row):
                # Equal-sized cells depend on available width, never on live text.
                strip.columnconfigure(column, weight=1, uniform='details')
                cell = tk.Frame(strip)
                cell.grid(row=0, column=column, sticky='ew', padx=(0, 8))
                tk.Label(cell, text=caption + ':', anchor='w', foreground='#526477').pack(side='left')
                value = tk.Label(cell, text='—', width=1, height=1, anchor='w',
                                 foreground='#173c68')
                value.pack(side='left', fill='x', expand=True)
                fields[key] = (value, HoverHint(value, '—'))
        self.refresh_control_details(fields)
        return fields

    def refresh_control_details(self, fields):
        data = {} if getattr(self, 'feedback', None) is None else self.feedback.snapshot()
        def xy(key):
            value = data.get(key)
            return '—' if value is None else ', '.join(f'{v:.2f}' for v in value)
        angles = data.get('sent_angles', '')
        sent = '—' if not angles else ', '.join(f'{angle:.1f}' for angle in angles)
        calculated = data.get('calculated_angles', '')
        calculated = '—' if not calculated else ', '.join(f'{angle:.1f}' for angle in calculated)
        def force(key):
            value = data.get(key)
            return '—' if value is None else ', '.join(f'{v*1e6:.3g}' for v in np.atleast_1d(value))
        speed = '—' if data.get('robot_speed_mm_s') is None else f"{data['robot_speed_mm_s']:.2f}"
        error = data.get('position_error_mm')
        error = '—' if error is None else f'{np.linalg.norm(error):.2f}'
        values = dict(
            desired=f"({xy('desired_xy_mm')})", measured=f"({xy('measured_xy_mm')})",
            u=f"({xy('correction_mm')})", commanded=f"({xy('commanded_xy_mm')})",
            ratio=str(data.get('ratio', '—')), angle=str(data.get('angle_deg', 'Auto')),
            speed=f"{speed} ({data.get('hold_speed_mm_s', '—')})",
            error=error, state=data.get('state', 'Control off'),
            integral=force('integral_force_n'), feedback_force=force('requested_force_n'),
            allocated_force=force('predicted_force_n'), residual=force('force_residual_n'),
            calculated=calculated, sent=sent)
        for key, text in values.items():
            widget, hint = fields[key]
            widget.config(text=text)
            # Full content remains accessible when a narrow window clips a cell.
            hint.text = text

    def point_signature(self):
        if not hasattr(self, 'learned'):
            return (hardware_case_name(self.calibration.control_case_var.get()), tuple(self.steps))
        setup = self.current_setup()
        _, corrections = self.learned_snapshot(setup)
        return (hardware_case_name(setup['control_case']), tuple(
            replace(step, target_xy_m=tuple(np.asarray(step.target_xy_m) + np.asarray(u)/1000))
            for step,u in zip(self.steps, corrections)))

    def learned_snapshot(self, setup):
        if not self.steps:
            raise ValueError('Load an XY sequence first.')
        signature = self.commander.hardware_signature(load_hardware_case(setup['control_case']))
        context = correction_context(setup, self.steps, signature, self.tracking_count_var.get())
        corrections = self.learned.snapshot(context, len(self.steps))
        if not self.use_learned_var.get():
            corrections = [[0.,0.] for step in self.steps]
        if getattr(self, 'manual_mode_var', None) is not None and self.manual_mode_var.get():
            corrections = [self.manual_targets.point(context, index) or correction
                           for index, correction in enumerate(corrections)]
        return context, corrections

    def change_manual_mode(self):
        if self.precalculating:
            self.manual_mode_var.set(getattr(self, 'manual_mode_active', False))
            self.status.config(text='Wait for precalculation to finish before switching mode.')
            return
        self.manual_mode_active = self.manual_mode_var.get()
        if self.feedback is not None:
            self.feedback.set_manual_mode(self.manual_mode_active)
        if self.manual_mode_var.get():
            self.control_requested = False
            self.control_button.config(text='Control off')
            if self.feedback is not None:
                self.feedback.enabled.clear()
            self.show_commanded_var.set(True)
            self.status.config(text='Manual: click the active point\'s commanded position to send an update while running. '
                               'When stopped, select a row, click, then Run selected point. Save u to keep it.')
        else:
            self.status.config(text='Manual off. Control on/off is available for the current point.')
        if not self.running:
            self.refresh_point_corrections()
        self.render()

    def set_manual_target(self, event, canvas):
        if self.precalculating:
            self.status.config(text='Wait for precalculation before changing the manual command.')
            return
        try:
            index = self.active_index
            if index is None:
                raise ValueError('Select a desired point in the table first.')
            transform = getattr(canvas, 'camera_transform', None)
            if transform is None or self.frame is None:
                raise ValueError('Wait for the camera image first.')
            ox, oy, scale = transform
            pixel = np.asarray([(event.x-ox)/scale, (event.y-oy)/scale])
            h, w = self.frame.shape[:2]
            if np.any(pixel < 0) or np.any(pixel >= [w, h]):
                raise ValueError('Click inside the camera image.')
            setup = self.setup if self.running else self.current_setup()
            axes = RobotAxes(setup['corners_px'], setup['rectangle_width_cm'],
                             setup['rectangle_height_cm'], setup['axis_samples_px'])
            target = axes.coordinates([pixel])[0] / 100
            self.validate_corrected_target(target, setup, allow_outside_pins=self.allow_extrapolation_var.get())
            context = self.run_correction_context if self.running else self.learned_snapshot(setup)[0]
            correction = target*1000 - np.asarray(self.steps[index].target_xy_m)*1000
            if self.running:
                if self.stop_event.is_set() or self.feedback is None:
                    raise ValueError('Wait for the run to finish stopping.')
                with self.arrival_gate.condition:
                    if self.arrival_gate.next_requested:
                        raise ValueError('The path is advancing; wait for the next active point.')
                    self.feedback.request_manual(self.steps[index], correction)
                    self.arrival_gate.condition.notify_all()
            self.manual_targets.remember(context, index, correction)
            self.precalculated_signature = None
            self.show_commanded_var.set(True)
            self.show_point_correction(index, correction)
            action = 'Queued for the active point; newest click replaces older pending clicks.' if self.running else 'Not sent. Run selected point to test.'
            self.status.config(text=f'Point {index+1}: manual command ({target[0]*1000:.2f}, {target[1]*1000:.2f}) mm. '
                               f'{action} Save u to keep the sent correction.')
            self.render()
        except (ValueError, TypeError, KeyError, ImportError) as error:
            self.status.config(text=f'Manual target not set: {error}')

    def save_learned(self):
        if self.precalculating:
            return
        try:
            context = (self.run_correction_context if self.running else
                       self.learned_snapshot(self.current_setup())[0])
            selection = self.table.selection()
            index = self.active_index if self.running else (int(selection[0]) if selection else self.active_index)
            if index is None:
                raise ValueError('Run a point and select it before saving its u.')
            if (getattr(self, 'feedback', None) is not None
                    and getattr(self.feedback, 'current_step', None) is self.steps[index]
                    and self.feedback.snapshot().get('force_compensated')):
                raise ValueError('The applied correction is in magnet inputs, not XY u. Its forces and angles are in the tracking CSV.')
            correction = self.latest_applied.point(context, index)
            if correction is None:
                raise ValueError('No command has been sent for this point in the current setup yet.')
            self.learned.remember(context, index, correction)
            if getattr(self, 'manual_mode_var', None) is not None and self.manual_mode_var.get():
                self.manual_targets.remember(context, index, correction)
            self.precalculated_signature = None
            self.show_point_correction(index, correction)
            self.render()
            self.status.config(text=f'Point {index + 1}: u ({correction[0]:.2f}, {correction[1]:.2f}) mm saved in memory. '
                               'Export points + u to keep the path on disk.')
        except (ValueError, TypeError, KeyError, ImportError) as error:
            show_view_dialog(self, messagebox.showerror, 'Save u', str(error))

    def export_learned(self):
        self.learned_file_action(False)

    def show_point_correction(self, index, correction):
        if not hasattr(self, 'display_corrections'):
            self.display_corrections = {}
        self.display_corrections[index] = list(correction)
        desired = np.asarray(self.steps[index].target_xy_m) * 1000
        u = np.asarray(correction, float)
        for key, value in zip(('ux', 'uy', 'cx', 'cy'), (*u, *(desired + u))):
            self.table.set(str(index), key, f'{value:.2f}')

    def refresh_point_corrections(self, setup=None):
        if not self.steps or not hasattr(self, 'learned'):
            return
        try:
            _, corrections = self.learned_snapshot(self.current_setup() if setup is None else setup)
        except (ValueError, TypeError, KeyError, ImportError):
            corrections = [[0., 0.] for step in self.steps]
        for index, correction in enumerate(corrections):
            self.show_point_correction(index, correction)

    def load_learned(self):
        self.learned_file_action(True)

    def learned_file_action(self, load):
        if (load and self.running) or self.precalculating:
            return
        try:
            context = (self.run_correction_context if self.running else
                       self.learned_snapshot(self.current_setup())[0])
            folder = XY_SEQUENCE_DIR.parent / 'setups'
            folder.mkdir(parents=True, exist_ok=True)
            options = dict(initialdir=str(folder), filetypes=[('Learned corrections', '*.json')])
            if load:
                path = show_view_dialog(self, filedialog.askopenfilename, title='Load learned u', **options)
            else:
                path = show_view_dialog(self, filedialog.asksaveasfilename, title='Export all points with saved u',
                                       initialfile=default_setup_name().replace('.json', '_u.json'),
                                       defaultextension='.json', **options)
            if path:
                if load:
                    self.learned.load(path, context, len(self.steps))
                    self.manual_targets.clear(context)
                    self.precalculated_signature = None
                    self.refresh_point_corrections()
                else:
                    self.learned.save(path, context, len(self.steps), steps=self.steps)
                self.status.config(text=f'{"Learned u loaded" if load else "Points and saved u exported"}: {path}')
        except (ValueError, TypeError, OSError, ImportError) as error:
            show_view_dialog(self, messagebox.showerror, 'Learned u', str(error))

    def clear_learned(self):
        if self.running or self.precalculating:
            return
        try:
            context, _ = self.learned_snapshot(self.current_setup())
            if show_view_dialog(self, messagebox.askyesno, 'Clear learned u', 'Clear learned corrections for this path/setup? Saved files are unchanged.'):
                self.learned.clear(context)
                self.manual_targets.clear(context)
                self.refresh_point_corrections()
                self.precalculated_signature = None
                self.status.config(text='Learned u cleared for this path/setup.')
        except (ValueError, ImportError) as error:
            show_view_dialog(self, messagebox.showerror, 'Learned u', str(error))

    def begin_crop_placement(self):
        if self.recording is not None:
            self.recording_status.config(text='Stop recording before changing the crop.')
            return
        self.crop_enabled_var.set(True)
        self.crop_fit_var.set(False)
        self.crop_placing = True
        self.recording_status.config(text='Click the camera for the export circle center; radius uses native pixels.')

    def recording_circle(self):
        if not self.crop_enabled_var.get() or self.frame is None:
            return None
        return video_crop_circle(self.frame.shape[1::-1], center=self.crop_center,
                                 radius=float(self.crop_radius_var.get()), fit=self.crop_fit_var.get(),
                                 corners=None if self.setup is None else self.setup['corners_px'])

    def precalculate_all(self):
        if self.running or self.precalculating:
            return
        try:
            if not self.steps:
                raise ValueError('Load an XY sequence first.')
            signature = self.point_signature()
            load_hardware_case(signature[0])
        except (ValueError, ImportError) as error:
            show_view_dialog(self, messagebox.showerror, 'Precalculate all', str(error))
            return
        self.precalculating = True
        self.refresh_next_button()
        self.set_running(True)
        self.precalculated_signature = None
        self.preparation_stop.clear()
        self.precalculate_button.config(state='disabled')
        self.run_button.config(state='disabled')
        self.run_selected_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.status.config(text='Precalculating all points in the background. No motors are moved; Stop cancels after the current calculation.')

        def worker():
            try:
                results = self.commander.prepare_points(signature[0], signature[1], active=lambda: not self.preparation_stop.is_set(),
                    report=lambda index, result: self.preparation_events.put(('point', index, result)))
                self.preparation_events.put(('done', signature, results))
            except Exception as error:
                self.preparation_events.put(('error', None, str(error)))
        threading.Thread(target=worker, daemon=True).start()
        self.preparation_after = self.parent.after(50, self.poll_precalculation)

    def poll_precalculation(self):
        self.preparation_after = None
        try:
            while True:
                kind, index, result = self.preparation_events.get_nowait()
                if kind == 'point':
                    if str(index) in self.table.get_children():
                        self.table.set(str(index), 'angles', ', '.join(f'{angle:.2f}' for angle in result['angles_deg']))
                        self.table.set(str(index), 'state', 'Precalculated')
                    self.status.config(text=f'Precalculated {index + 1} points. No motors moved.')
                else:
                    self.precalculating = False
                    self.refresh_next_button()
                    self.set_running(False)
                    self.precalculate_button.config(state='normal')
                    self.run_button.config(state='normal')
                    self.run_selected_button.config(state='normal')
                    self.stop_button.config(state='disabled')
                    if kind == 'error':
                        self.status.config(text=f'Precalculation failed: {result}. Partial results will not be used.')
                    elif result is not None and not self.preparation_stop.is_set() and index == self.point_signature():
                        self.precalculated_signature = index
                        self.status.config(text='All points precalculated. Run selected / Run all will reuse these angles; arrival and Next checks still apply.')
                    else:
                        self.status.config(text='Precalculation cancelled or settings changed. Precalculate all again.')
                    if self.precalculated_signature is None:
                        for row in self.table.get_children():
                            if self.table.set(row, 'state') == 'Precalculated':
                                self.table.set(row, 'state', 'Not ready')
        except queue.Empty:
            pass
        if self.precalculating:
            self.preparation_after = self.parent.after(50, self.poll_precalculation)

    def run(self, *, selected_only=False):
        if self.running:
            return
        if getattr(self, 'precalculating', False):
            self.status.config(text='Wait for precalculation to finish, or press Stop.')
            return
        prepared = False
        try:
            manual = getattr(self, 'manual_mode_var', None) is not None and self.manual_mode_var.get()
            allow_extrapolation = self.allow_extrapolation_var.get()
            if getattr(self, 'precalculated_signature', None) is not None:
                if self.precalculated_signature != self.point_signature():
                    self.precalculated_signature = None  # Changed learned u must not reuse old motor angles.
                else:
                    prepared = True
            arrival_gate = ArrivalGate(float(self.arrival_tolerance_var.get()), float(self.arrival_hold_var.get()), require_arrival=False)
            feedback = None
            if hasattr(self, 'force_step_var'):
                def validate_correction(target):
                    self.validate_corrected_target(target, setup, allow_outside_pins=allow_extrapolation)
                pid = None
                angle_rate, deadband = 20., .1
                if hasattr(self, 'force_pid_settings'):
                    values = {key: float(variable.get()) for key, variable in self.force_pid_settings.items()}
                    pid = MagneticForcePID(kp=values['Kp']*1e-3, ki=values['Ki']*1e-3, kv=values['Kv']*1e-3,
                        velocity_tau_s=values['Velocity filter s'], integral_limit_n=values['I max µN']*1e-6,
                        period_s=values['Period s'])
                    angle_rate, deadband = values['Rate deg/s'], values['Deadband deg']
                feedback = ForcePIDFeedback(validate=validate_correction, manual=manual,
                    max_angle_step_deg=float(self.force_step_var.get()), preview_only=self.force_preview_var.get(),
                    pid=pid, angle_rate_deg_s=angle_rate, motor_deadband_deg=deadband,
                    hold_speed_mm_s=float(self.force_hold_speed_var.get()) if hasattr(self, 'force_hold_speed_var') else .5)
                if self.control_requested and not manual:
                    feedback.enabled.set()
            setup, pixels = self.prepare(validate=False)
            learned_context, corrections = (self.learned_snapshot(setup) if hasattr(self, 'learned') else
                                            (None, [[0.,0.] for step in self.steps]))
            if selected_only:
                selection = self.table.selection()
                if not selection:
                    raise ValueError("Select a point in the table or click its marker first.")
                indices = (int(selection[0]),)
            else:
                indices = tuple(range(len(self.steps)))
            validate_projected_points(self.steps, setup, pixels, indices=indices,
                                      allow_outside_pins=allow_extrapolation)
            for index in indices:
                if np.linalg.norm(corrections[index]) > 0:
                    target = np.asarray(self.steps[index].target_xy_m) + np.asarray(corrections[index])/1000
                    self.validate_corrected_target(target, setup, allow_outside_pins=allow_extrapolation)
            if not self.camera_running() or self.frame_received_at is None or time.monotonic() - self.frame_received_at > 2:
                raise ValueError("Connect the live camera and wait for a fresh frame before running.")
            port = self.commander.port_provider()
            if port is None or not getattr(port, "is_open", False):
                raise ValueError("Connect the serial controller first.")
            load_hardware_case(setup["control_case"])
            self.configure_tracking(setup)
            if self.tracking_tracker is None or not self.tracking_enabled:
                raise ValueError('Place the Phase 3 ROI and detect the robots before running.')
            if hasattr(self, 'command_roi') and (self.command_roi is None or self.tracking_group_xy is None
                    or len(self.tracking_detections) != int(self.tracking_count_var.get())):
                raise ValueError('Wait for a fresh, complete robot group inside the Phase 3 ROI before running.')
        except (ValueError, TypeError, KeyError, ImportError) as error:
            show_view_dialog(self, messagebox.showerror, "Cannot run sequence", str(error))
            return
        self.setup, self.pixels = setup, pixels
        self.running = True
        self.run_manual_mode = manual
        self.active_index = None
        self.stop_event.clear()
        self.job += 1
        job = self.job
        steps = tuple(self.steps[index] for index in indices)
        self.run_indices = indices
        self.arrival_gate = arrival_gate
        self.feedback = feedback
        self.run_correction_context = learned_context
        for index, correction in enumerate(corrections):
            self.show_point_correction(index, correction)
        for widget in getattr(self, 'learned_controls', []):
            widget.config(state='disabled')
        for entry in getattr(self, 'force_entries', []):
            entry.config(state='disabled')
        if hasattr(self, 'precalculate_button'):
            self.precalculate_button.config(state='disabled')
        self.set_running(True)
        for button in (self.load_button, self.run_button, self.run_selected_button):
            button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.units_combo.config(state="disabled")
        self.extrapolation_check.config(state="disabled")
        self.refresh_next_button()
        self.arrival_tolerance_entry.config(state="disabled")
        self.arrival_hold_entry.config(state="disabled")
        for index in indices:
            self.table.set(str(index), "state", "Pending")
            self.table.set(str(index), "angles", "")
        self.status.config(text="Sequence started. Stop prevents subsequent poses; the last sent pose remains applied.")

        def active():
            return (not self.stop_event.is_set() and job == self.job
                    and self.commander.port_provider() is port and getattr(port, "is_open", False)
                    and self.camera_running() and self.frame_received_at is not None
                    and time.monotonic() - self.frame_received_at <= 2)

        def remember_correction(index, u):
            if learned_context is not None:
                self.latest_applied.remember(learned_context, indices[index], u)

        def worker():
            try:
                outcome = execute_xy_sequence(steps, self.commander, hardware_case_name(setup["control_case"]),
                                              active=active, report=lambda kind, index, result:
                                              self.events.put((job, kind, indices[index], result)),
                                              arrival_gate=arrival_gate, prepared=prepared, feedback=feedback,
                                              initial_corrections=[corrections[index] for index in indices],
                                              on_correction=remember_correction)
                self.events.put((job, outcome, None, None))
            except Exception as error:
                self.events.put((job, "error", None, str(error)))

        threading.Thread(target=worker, daemon=True).start()
        self.after_id = self.parent.after(50, self.poll)

    def stop(self):
        if getattr(self, 'precalculating', False):
            self.preparation_stop.set()
            self.status.config(text='Precalculation cancellation requested; waiting for the current solver to return. No motors moved.')
            return
        if self.running:
            self.stop_event.set()
            self.stop_button.config(state="disabled")
            self.next_button.config(state="disabled")
            self.status.config(text="Stop requested. Unsent commands will be cancelled; an ongoing calculation may take time to return. Last pose stays applied.")

    def next_point(self):
        if not self.running:
            if not self.steps or self.precalculating:
                return
            selection = self.table.selection()
            current = int(selection[0]) if selection else self.active_index
            index = 0 if current is None else (current + 1) % len(self.steps)
            self.table.selection_set(str(index))
            self.table.see(str(index))
            self.select_point()
            self.run_selected()
            return
        if self.running and self.arrival_gate is not None and self.arrival_gate.request_next():
            self.next_button.config(state="disabled")
            self.status.config(text='Next requested. Advancing after the current row minimum wait; no arrival requirement or extra queued presses.')
        else:
            self.status.config(text='A Next press is already queued, or the run is stopping.')

    def refresh_next_button(self):
        if not self.running:
            selection = self.table.selection() if self.steps else ()
            current = int(selection[0]) if selection else self.active_index
            last = bool(self.steps) and current == len(self.steps) - 1
            self.next_button.config(state='normal' if self.steps and not getattr(self, 'precalculating', False) else 'disabled',
                                    text='First point' if last else 'Next point')
            return
        gate = self.arrival_gate
        waiting = self.running and gate is not None and gate.waiting_for_next
        ready = gate is not None and (not gate.require_arrival or (waiting and gate.ready)) and not self.stop_event.is_set() and not gate.next_requested
        last = self.active_index is not None and self.active_index == self.run_indices[-1] if self.running else False
        self.next_button.config(state="normal" if ready else "disabled", text="Finish" if last else "Next point")
        if waiting and self.active_index is not None and not self.stop_event.is_set():
            self.table.set(str(self.active_index), "state", "Await Next" if ready else 'Next queued' if gate.next_requested else "Arriving")
            self.status.config(text=(
                f"Point {self.active_index + 1} sent. Press {'Finish' if last else 'Next'} when you choose; arrival is not required."
                if ready else 'Next queued; waiting for the current row minimum wait to finish.' if gate.next_requested else 'Waiting for the motor command.'
            ))

    def poll(self):
        self.after_id = None
        try:
            while True:
                job, kind, index, result = self.events.get_nowait()
                if job != self.job:
                    continue
                if kind == 'corrected':
                    self.table.set(str(index), 'angles', ', '.join(f'{angle:.2f}' for angle in result['angles_deg']))
                    continue
                if kind in ("calculating", "sent", "completed", "waiting_arrival", "arrived"):
                    self.active_index = index
                    state = {"calculating": "Calculating", "sent": "Moving", "completed": "Completed",
                             "waiting_arrival": "Arriving", "arrived": "Await Next"}[kind]
                    self.table.set(str(index), "state", state)
                    self.table.see(str(index))
                    if result is not None:
                        self.table.set(str(index), "angles", ", ".join(f"{angle:.2f}" for angle in result["angles_deg"]))
                    target = self.steps[index].target_xy_m
                    self.status.config(text=f"Point {index + 1}/{len(self.steps)}: {state.lower()}; desired XY ({target[0] * 1000:g}, {target[1] * 1000:g}) mm.")
                    self.render()
                    self.refresh_next_button()
                else:
                    self.running = False
                    for widget in getattr(self, 'learned_controls', []):
                        widget.config(state='normal')
                    for entry in getattr(self, 'force_entries', []):
                        entry.config(state='normal')
                    if getattr(self, 'feedback', None) is not None:
                        self.feedback.enabled.clear()
                    if hasattr(self, 'precalculate_button'):
                        self.precalculate_button.config(state='normal')
                    self.set_running(False)
                    for button in (self.load_button, self.run_button, self.run_selected_button):
                        button.config(state="normal")
                    self.stop_button.config(state="disabled")
                    self.units_combo.config(state="readonly")
                    self.extrapolation_check.config(state="normal")
                    self.arrival_gate = None
                    self.refresh_next_button()
                    self.arrival_tolerance_entry.config(state="normal")
                    self.arrival_hold_entry.config(state="normal")
                    if kind != "finished" and self.active_index is not None:
                        self.table.set(str(self.active_index), "state", "Failed" if kind == "error" else "Stopped")
                    self.status.config(text=("Sequence complete." if kind == "finished" else
                                             f"Sequence failed: {result}. No further points sent." if kind == "error" else
                                             "Sequence stopped (Stop, camera loss, or controller disconnect). Last sent pose remains applied."))
                    if kind == "error":
                        show_view_dialog(self, messagebox.showerror, "Sequence stopped", str(result))
        except queue.Empty:
            pass
        if self.running:
            self.after_id = self.parent.after(50, self.poll)

    def commanded_pixel(self):
        index = self.active_index
        if self.setup is None or index is None or index >= len(self.steps):
            return None
        target = np.asarray(self.steps[index].target_xy_m) * 1000
        correction = getattr(self, 'display_corrections', {}).get(index, [0., 0.])
        commanded = target + correction
        if self.running and self.feedback is not None:
            data = self.feedback.snapshot()
            if data.get('force_compensated'):
                return None  # Direct magnet-force corrections have no unique commanded XY.
            if data.get('desired_xy_mm') is not None and np.allclose(data['desired_xy_mm'], target):
                if data.get('commanded_xy_mm') is not None:
                    commanded = np.asarray(data['commanded_xy_mm'])
        axes = RobotAxes(self.setup['corners_px'], self.setup['rectangle_width_cm'],
                         self.setup['rectangle_height_cm'], self.setup['axis_samples_px'])
        return axes.pixel_positions([commanded/10])[0]

    def render(self, canvas=None):
        if canvas is None:
            self.render_revision = getattr(self, 'render_revision', 0) + 1
        popup = getattr(self, 'camera_popout', None)
        if canvas is None and popup is not None and popup.is_open:
            self.screen_points = None
            self.canvas.delete('all')
            self.canvas.create_text(max(1, self.canvas.winfo_width()) / 2, max(1, self.canvas.winfo_height()) / 2,
                                    text='Camera is displayed in the pop-out window', fill='#c5d9ee')
            return
        main_view = canvas is None
        canvas = self.canvas if main_view else canvas
        canvas.delete("all")
        canvas.screen_points = None
        if self.frame is None:
            return
        h, w = self.frame.shape[:2]
        cw, ch = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
        low, high = np.array([0., 0.]), np.array([float(w), float(h)])
        if self.pixels is not None:
            low = np.minimum(low, self.pixels.min(axis=0) - [15, 15])
            high = np.maximum(high, self.pixels.max(axis=0) + [15, 15])
        extent = high - low
        scale = min(cw / extent[0], ch / extent[1])
        dw, dh = max(1, round(w * scale)), max(1, round(h * scale))
        ox, oy = (np.array([cw, ch]) - extent * scale) / 2 - low * scale
        canvas.camera_photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)).resize((dw, dh)))
        if main_view:
            self.photo = canvas.camera_photo
        canvas.create_image(ox + dw / 2, oy + dh / 2, image=canvas.camera_photo, anchor="center")
        canvas.camera_transform = (ox, oy, scale)
        if hasattr(self, 'crop_enabled_var'):
            try:
                circle = (self.recording.crop if self.recording is not None else None)
                circle = (circle.x,circle.y,circle.radius) if circle is not None else self.recording_circle()
                if circle is not None:
                    x,y = np.asarray(circle[:2])*scale + [ox,oy]
                    radius = circle[2]*scale
                    canvas.create_oval(x-radius,y-radius,x+radius,y+radius, outline='#66ccff', width=1, dash=(3,6))
            except (ValueError, TypeError):
                pass  # Export validates inputs; invalid edits must not stop the live view.
        if self.setup is None:
            return
        if len(self.setup['corners_px']) == 4:
            corners = np.asarray(self.setup["corners_px"]) * scale + [ox, oy]
            canvas.create_polygon(*corners.ravel(), fill="", outline="#995599", width=1, dash=(2, 10))

        def enabled(name):
            variable = getattr(self, name, None)
            return variable is not None and variable.get()

        if self.pixels is not None:
            points = self.pixels * scale + [ox, oy]
            canvas.screen_points = points
            if main_view:
                self.screen_points = points
            if len(points) > 1 and enabled("show_path_var"):
                canvas.create_line(*points.ravel(), fill="#338888", width=1, dash=(2, 8))
            outside, _ = point_limits(self.pixels, self.setup)
            for index, (x, y) in enumerate(points):
                canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill="", outline="#ff4444" if outside[index] else "#00cccc", width=1)
            if self.active_index is not None and self.active_index < len(points):
                x, y = points[self.active_index]
                canvas.create_oval(x - 3, y - 3, x + 3, y + 3, outline="#ff9900", width=1)
                canvas.create_line(x - 3, y, x + 3, y, fill="#ff9900", width=1)
                canvas.create_line(x, y - 3, x, y + 3, fill="#ff9900", width=1)
                canvas.create_text(x + 5, y - 5, text=f"P{self.active_index + 1}", fill="#ff9900", font=("Segoe UI", 7), anchor="sw")
        if enabled('show_commanded_var'):
            try:
                commanded = self.commanded_pixel()
                if commanded is not None:
                    x, y = commanded * scale + [ox, oy]
                    canvas.create_polygon(x, y-4, x+4, y, x, y+4, x-4, y,
                                          outline='#ff55dd', fill='', width=1)
                    canvas.create_text(x+6, y+5, text='C', fill='#ff55dd', font=('Segoe UI', 8), anchor='nw')
            except (ValueError, TypeError, KeyError):
                pass
        tracker = getattr(self, "tracking_tracker", None)
        if tracker is not None and enabled("show_roi_var"):
            x, y = tracker.center * scale + [ox, oy]
            radius = tracker.radius * scale
            canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                                    outline="#449944", width=1, dash=(2, 10))
        if enabled("show_trails_var"):
            for trail in self.robot_trails.values():
                segment = []
                for point in [*trail, None]:
                    if point is not None:
                        segment.append(np.asarray(point) * scale + [ox, oy])
                    else:
                        if len(segment) > 1:
                            canvas.create_line(*np.asarray(segment).ravel(), fill="#449944", width=1)
                        segment = []
        for item in getattr(self, "tracking_detections", []):
            x, y = np.asarray(item["center"]) * scale + [ox, oy]
            canvas.create_line(x - 2, y, x + 2, y, fill="#00ff66", width=1)
            canvas.create_line(x, y - 2, x, y + 2, fill="#00ff66", width=1)
            canvas.create_text(x + 4, y + 4, text=f"R{item['robot_id']}", fill="#00ff66", font=("Segoe UI", 7), anchor="nw")

    def close(self):
        if getattr(self, 'diagnostics', None) is not None:
            self.stop_diagnostics()
        if hasattr(self, 'help_hint'):
            self.help_hint.hide()
        if hasattr(self, 'preparation_stop'):
            self.preparation_stop.set()
        if getattr(self, 'preparation_after', None) is not None:
            self.parent.after_cancel(self.preparation_after)
            self.preparation_after = None
        if getattr(self, 'camera_popout', None) is not None:
            self.camera_popout.close()
        if getattr(self, 'recording', None) is not None:
            self.stop_recording()
        self.stop_event.set()
        self.job += 1
        if self.after_id is not None:
            self.parent.after_cancel(self.after_id)
            self.after_id = None
