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
from command_recording import CommandRecording, draw_command_overlay
from camera_popout import CameraPopout, show_view_dialog


class CommandTab:
    def __init__(self, parent, calibration, commander, camera_running, *, set_running=lambda value: None,
                 visible=lambda: True):
        self.parent, self.calibration, self.commander = parent, calibration, commander
        self.camera_running, self.set_running, self.visible = camera_running, set_running, visible
        self.steps, self.pixels, self.setup = [], None, None
        self.sequence_path, self.input_units = None, None
        self.frame, self.frame_received_at = None, None
        self.running = False
        self.stop_event = threading.Event()
        self.events = queue.Queue()
        self.job = 0
        self.active_index = None
        self.show_all = True
        self.after_id = None
        self.arrival_gate = None
        self.cell_editor = None
        self.render_revision = 0
        self.camera_popout = None
        self.recording = None
        self.recording_fps_var = tk.StringVar(value='5')
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
        tk.Label(camera_view_bar, text='Live processing FPS', background='#edf4ff', foreground='#173c68').pack(side='left')
        tk.Spinbox(camera_view_bar, from_=1, to=15, textvariable=self.calibration.live_fps_var, width=3).pack(side='left', padx=4)
        tk.Label(camera_view_bar, text='Opens the larger camera window; all controls remain here.',
                 background='#edf4ff', foreground='#173c68').pack(side='left', padx=4)
        bar = tk.Frame(parent)
        bar.pack(fill="x", padx=8, pady=6)
        self.load_button = tk.Button(bar, text="Load XY sequence", command=self.load_sequence)
        self.load_button.pack(side="left", padx=3)
        tk.Button(bar, text="Export XY sequence", command=self.export_sequence).pack(side="left", padx=3)
        self.calibration_button = tk.Button(bar, text="Use current calibration", command=lambda: self.refresh_calibration(reset_tracking=True))
        self.calibration_button.pack(side="left", padx=3)
        actions = tk.Frame(parent)
        actions.pack(fill="x", padx=8, pady=3)
        self.show_button = tk.Button(actions, text="Show all points", command=self.show_points)
        self.show_button.pack(side="left", padx=3)
        self.run_selected_button = tk.Button(actions, text="Run selected point", command=self.run_selected)
        self.run_selected_button.pack(side="left", padx=3)
        self.run_button = tk.Button(actions, text="Run all", command=self.run)
        self.run_button.pack(side="left", padx=3)
        self.stop_button = tk.Button(actions, text="Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=3)
        self.next_button = tk.Button(actions, text="Next point", command=self.next_point, state="disabled")
        self.next_button.pack(side="left", padx=3)
        self.allow_extrapolation_var = tk.BooleanVar(value=False)
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
        tk.Label(parent, anchor="w", justify="left", wraplength=1000, text=(
            "Calibration sizes: cm. Sequence: select mm, cm or m. Auto reads the unit header; without a header it uses mm. "
            "Table shows mm. Double-click Ratio, Angle, Wait or Zero to edit each point (Enter applies). "
            "Ratio defaults to 1; Angle Auto is unconstrained and is ignored for ratio 1. "
            "Each point is calculated fresh and sent. After the row's minimum wait, all robots must reach the target and stay within tolerance. "
            "Press Next point to advance (Finish for the last point). There is no automatic progression. "
            "Stop prevents further commands; it does not undo the last motor pose."
        )).pack(fill="x", padx=10, pady=3)
        table_frame = tk.Frame(parent)
        table_frame.pack(fill="x", padx=8)
        self.table = ttk.Treeview(table_frame, columns=("x", "y", "wait", "zero", "ratio", "angle", "state", "angles"),
                                  show="tree headings", height=6, selectmode="browse")
        self.table.tag_configure("outside", foreground="#b00020")
        self.table.heading("#0", text="Point")
        self.table.column("#0", width=55, stretch=False)
        for key, title, width in (("x", "X (mm)", 75), ("y", "Y (mm)", 75),
                                  ("wait", "Wait (s)", 65), ("zero", "Zero (deg)", 75),
                                  ("ratio", "Ratio", 60), ("angle", "Angle (deg)", 80),
                                  ("state", "State", 95), ("angles", "Calculated motor angles (deg)", 380)):
            self.table.heading(key, text=title)
            self.table.column(key, width=width, stretch=(key == "angles"))
        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
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
        tk.Button(tracking_bar, text="Start tracking", command=self.start_tracking).pack(side="left", padx=2)
        tk.Button(tracking_bar, text="Stop tracking", command=self.stop_tracking).pack(side="left", padx=2)
        tk.Button(tracking_bar, text="Export tracking CSV", command=self.export_tracking).pack(side="left", padx=2)
        for label, variable in (("Phase 3 robots", self.tracking_count_var), ("Min area", self.tracking_area_var),
                                ("Contrast", self.tracking_contrast_var), ("ROI radius (px)", self.tracking_radius_var)):
            tk.Label(tracking_bar, text=label).pack(side="left", padx=(7, 2))
            tk.Entry(tracking_bar, textvariable=variable, width=4).pack(side="left")
        options = tk.Frame(parent)
        options.pack(fill="x", padx=8)
        for label, variable in (("Show desired lines", self.show_path_var), ("Show moving ROI", self.show_roi_var),
                                ("Show robot trails", self.show_trails_var)):
            tk.Checkbutton(options, text=label, variable=variable, command=self.render).pack(side="left", padx=3)
        tk.Label(options, text="Arrival tolerance (mm)").pack(side="left", padx=(8, 2))
        self.arrival_tolerance_entry = tk.Entry(options, textvariable=self.arrival_tolerance_var, width=4)
        self.arrival_tolerance_entry.pack(side="left")
        tk.Label(options, text="In-range hold (s)").pack(side="left", padx=(8, 2))
        self.arrival_hold_entry = tk.Entry(options, textvariable=self.arrival_hold_var, width=4)
        self.arrival_hold_entry.pack(side="left")
        self.tracking_status = tk.Label(parent, text="Tracking starts automatically when calibration is applied.", anchor="w", wraplength=1000)
        self.tracking_status.pack(fill="x", padx=10)
        recording_bar = tk.Frame(parent)
        recording_bar.pack(fill='x', padx=8, pady=3)
        tk.Button(recording_bar, text='Start recording', command=self.start_recording).pack(side='left', padx=3)
        self.recording_stop_button = tk.Button(recording_bar, text='Stop / save recording', command=self.stop_recording, state='disabled')
        self.recording_stop_button.pack(side='left', padx=3)
        tk.Label(recording_bar, text='Video FPS').pack(side='left', padx=3)
        tk.Entry(recording_bar, textvariable=self.recording_fps_var, width=4).pack(side='left')
        self.recording_status = tk.Label(recording_bar, text='Camera + annotated + blue-screen overlays + tracking CSV', anchor='w')
        self.recording_status.pack(side='left', padx=8)
        tk.Label(parent, text="Green: tracked robots    Cyan: desired points    Orange: active target    Red: beyond pins. Select a row or click a target.",
                 anchor="w").pack(fill="x", padx=10)
        self.canvas = tk.Canvas(parent, background="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=(3, 8))
        self.canvas.bind("<Configure>", lambda event: self.render())
        self.canvas.bind("<Button-1>", self.click_target)

    def load_sequence(self):
        if self.running:
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
        if self.running or self.sequence_path is None:
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
                f"{step.wait_seconds:g}", f"{step.zero_angle_deg:g}", f"{step.ratio:g}",
                "Auto" if step.angle_deg is None else f"{step.angle_deg:g}", "Pending", ""))
        self.refresh_calibration()

    def edit_point_cell(self, event):
        if self.running:
            return
        row = self.table.identify_row(event.y)
        column = self.table.identify_column(event.x)
        fields = {"#3": "wait", "#4": "zero", "#5": "ratio", "#6": "angle"}
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
        if self.running:
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
            self.status.config(text=f'Point {index + 1} settings applied. Export XY sequence to save them.')
        except ValueError as error:
            self.status.config(text=f'Point edit rejected: {error}')

    def current_setup(self):
        setup = json.loads(json.dumps(self.calibration.setup_data()))
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

    def refresh_calibration(self, *, reset_tracking=False):
        if self.running:
            return
        try:
            setup = self.current_setup()
            self.configure_tracking(setup, reset=reset_tracking)
            if not self.steps:
                self.setup, self.pixels = setup, None
                self.status.config(text="Calibration applied; live tracking enabled. Load an XY sequence to show desired points.")
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
        self.render()

    def accept_live_frame(self, frame):
        self.frame = frame
        self.frame_received_at = time.monotonic()
        if self.setup is not None and list(frame.shape[1::-1]) != self.setup["video_size_px"]:
            self.stop_event.set()
            self.pixels = None
            self.tracking_tracker = None
            self.tracking_detections = []
            self.tracking_group_xy = None
            self.tracking_status.config(text="Tracking stopped: camera size no longer matches calibration.")
            self.status.config(text="Camera resolution changed. Sequence stopped; load matching calibration before running again.")
        first_row = len(self.tracking_rows)
        if getattr(self, "tracking_tracker", None) is not None:
            self.process_tracking_frame()
        if getattr(self, 'recording', None) is not None:
            try:
                options = dict(show_path=self.show_path_var.get(), show_roi=self.show_roi_var.get(),
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
                self.recording_status.config(text=f'Recording {self.recording.frame_count} frames; source-resolution videos + CSV.')
            except (ValueError, OSError, cv2.error) as error:
                self.stop_recording()
                self.recording_status.config(text=f'Recording stopped: {error}. Partial files kept.')
        if self.visible():
            self.render()

    def configure_tracking(self, setup, *, reset=False):
        if setup.get("coordinate_system") != "robot_axes":
            raise ValueError("Capture Center, +X and +Y, or load a complete robot-axis calibration first.")
        signature = json.dumps({key: setup.get(key) for key in (
            "corners_px", "rectangle_width_cm", "rectangle_height_cm", "axis_samples_px", "video_size_px")}, sort_keys=True)
        if not reset and signature == self.tracking_signature:
            if self.tracking_enabled and self.tracking_tracker is None:
                self.start_tracking(setup=setup)
            return
        self.tracking_signature = signature
        self.setup = setup
        self.tracking_area_var.set(str(setup["minimum_area"]))
        self.tracking_contrast_var.set(str(setup.get("dark_contrast_threshold", 10)))
        self.tracking_radius_var.set(str(setup["roi_circle_px"][2]))
        self.tracking_tracker = None
        self.tracking_detections = []
        self.tracking_group_xy = None
        if self.tracking_enabled:
            self.start_tracking(setup=setup)

    def new_tracking_engine(self, setup, center=None):
        count = int(self.tracking_count_var.get())
        area, contrast = float(self.tracking_area_var.get()), float(self.tracking_contrast_var.get())
        radius = float(self.tracking_radius_var.get())
        circle = setup["roi_circle_px"]
        center = circle[:2] if center is None else center
        return MovingROICalibration(
            setup["corners_px"], setup["rectangle_width_cm"], setup["rectangle_height_cm"],
            (*center, radius), count, area, setup["axis_samples_px"],
            contrast_threshold=contrast, limit_to_pins=False,
        )

    def start_tracking(self, *, setup=None):
        if getattr(self, "arrival_gate", None) is not None:
            self.arrival_gate.observe(None)
            self.refresh_next_button()
        try:
            setup = self.current_setup() if setup is None else setup
            if setup.get("coordinate_system") != "robot_axes":
                raise ValueError("Complete robot XY axes before tracking in Commands.")
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
        except (ValueError, TypeError) as error:
            self.tracking_detections, self.tracking_group_xy = [], None
            if getattr(self, "arrival_gate", None) is not None:
                self.arrival_gate.observe(None)
                self.refresh_next_button()
            self.tracking_status.config(text=f"Tracking unavailable: {error}")
            return
        if getattr(self, "arrival_gate", None) is not None:
            positions = (None if center is None else
                         np.asarray([item["xy_cm"] for item in self.tracking_detections]) * .01)
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
        for item in self.tracking_detections or [None]:
            self.tracking_rows.append({
                "frame": self.tracking_frame_number, "time_s": elapsed,
                "robot_id": "" if item is None else item["robot_id"],
                "robot_x_px": "" if item is None else item["center"][0],
                "robot_y_px": "" if item is None else item["center"][1],
                "robot_x_mm": "" if item is None else item["xy_cm"][0] * 10,
                "robot_y_mm": "" if item is None else item["xy_cm"][1] * 10,
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
            })
        self.tracking_frame_number += 1
        detail = "Group missing; ROI held at last complete detection."
        if group_mm is not None:
            detail = f"Group XY ({group_mm[0]:.2f}, {group_mm[1]:.2f}) mm."
        if error_mm is not None:
            detail += f" Active-target error {error_mm:.2f} mm."
        if gate is not None and gate.error_mm is not None:
            detail += f" Arrival maximum robot error {gate.error_mm:.2f} mm (limit {gate.tolerance_mm:g} mm)."
        self.tracking_status.config(text=f"Tracking {len(self.tracking_detections)}/{tracker.count} robots. {detail}")

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
            self.recording = CommandRecording(path, self.frame.shape[1::-1], float(self.recording_fps_var.get()))
            self.recording_stop_button.config(state='normal')
            self.recording_status.config(text='Recording started. Stop / save finalizes all four files.')
        except (ValueError, OSError, cv2.error) as error:
            show_view_dialog(self, messagebox.showerror, 'Recording', str(error))

    def stop_recording(self):
        if self.recording is not None:
            recording, self.recording = self.recording, None
            recording.close()
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

    def show_points(self):
        self.show_all = True
        if self.running:
            self.render()
        else:
            self.refresh_calibration()

    def open_camera_popout(self):
        if self.camera_popout is not None and self.camera_popout.is_open:
            self.camera_popout.show()
            return
        self.camera_popout = CameraPopout(
            self.parent, 'GUI005 — Phase 3 camera', self.render,
            actions=(('Show all points', self.show_points, None),
                     ('Next point', self.next_point, self.next_button),
                     ('Stop', self.stop, self.stop_button)),
            render_version=lambda: (self.render_revision, self.frame_received_at),
            on_close=self.render,
            status=lambda: f"{self.status.cget('text')}  {self.tracking_status.cget('text')}",
        )
        canvas = self.camera_popout.canvas
        canvas.bind('<Button-1>', lambda event: self.click_target(event, screen_points=getattr(canvas, 'screen_points', None)))
        self.render()

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

    def run(self, *, selected_only=False):
        if self.running:
            return
        try:
            arrival_gate = ArrivalGate(float(self.arrival_tolerance_var.get()), float(self.arrival_hold_var.get()))
            setup, pixels = self.prepare(validate=False)
            if selected_only:
                selection = self.table.selection()
                if not selection:
                    raise ValueError("Select a point in the table or click its marker first.")
                indices = (int(selection[0]),)
            else:
                indices = tuple(range(len(self.steps)))
            allow_extrapolation = self.allow_extrapolation_var.get()
            validate_projected_points(self.steps, setup, pixels, indices=indices,
                                      allow_outside_pins=allow_extrapolation)
            if not self.camera_running() or self.frame_received_at is None or time.monotonic() - self.frame_received_at > 2:
                raise ValueError("Connect the live camera and wait for a fresh frame before running.")
            port = self.commander.port_provider()
            if port is None or not getattr(port, "is_open", False):
                raise ValueError("Connect the serial controller first.")
            load_hardware_case(setup["control_case"])
            self.configure_tracking(setup)
            if self.tracking_tracker is None or not self.tracking_enabled:
                raise ValueError("Start robot tracking before running an arrival-controlled sequence.")
        except (ValueError, TypeError, KeyError, ImportError) as error:
            show_view_dialog(self, messagebox.showerror, "Cannot run sequence", str(error))
            return
        outside, _ = point_limits(pixels, setup)
        extrapolated = sum(bool(outside[index]) for index in indices)
        description = f"selected point {indices[0] + 1}" if selected_only else f"all {len(indices)} points"
        warning = f"\nWARNING: {extrapolated} target(s) beyond pins. Mapping is extrapolated; confirm they are physically safe." if extrapolated else ""
        if not show_view_dialog(self, messagebox.askyesno, "Run XY targets", f"Run {description} using {setup['control_case']}?\nInput units: {self.input_units}; table: mm; calibration: cm.\nArrival: all tracked robots within {arrival_gate.tolerance_mm:g} mm for {arrival_gate.stable_seconds:g} s; then press Next.\nRow wait is the minimum hold after sending; no automatic advance.\nConfirm the units and physical hardware match.{warning}"):
            return
        self.setup, self.pixels = setup, pixels
        self.running = True
        self.active_index = None
        self.stop_event.clear()
        self.job += 1
        job = self.job
        steps = tuple(self.steps[index] for index in indices)
        self.run_indices = indices
        self.arrival_gate = arrival_gate
        self.set_running(True)
        for button in (self.load_button, self.calibration_button, self.run_button, self.run_selected_button):
            button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.units_combo.config(state="disabled")
        self.extrapolation_check.config(state="disabled")
        self.next_button.config(state="disabled", text="Next point")
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

        def worker():
            try:
                outcome = execute_xy_sequence(steps, self.commander, hardware_case_name(setup["control_case"]),
                                              active=active, report=lambda kind, index, result:
                                              self.events.put((job, kind, indices[index], result)),
                                              arrival_gate=arrival_gate)
                self.events.put((job, outcome, None, None))
            except Exception as error:
                self.events.put((job, "error", None, str(error)))

        threading.Thread(target=worker, daemon=True).start()
        self.after_id = self.parent.after(50, self.poll)

    def stop(self):
        if self.running:
            self.stop_event.set()
            self.stop_button.config(state="disabled")
            self.next_button.config(state="disabled")
            self.status.config(text="Stop requested. Unsent commands will be cancelled; an ongoing calculation may take time to return. Last pose stays applied.")

    def next_point(self):
        if self.running and self.arrival_gate is not None and self.arrival_gate.request_next():
            self.next_button.config(state="disabled")
            self.status.config(text="Progression confirmed. Advancing only the current step; no queued Next presses.")
        else:
            self.status.config(text="Next is unavailable: wait for a fresh, complete robot detection inside the arrival tolerance.")

    def refresh_next_button(self):
        gate = self.arrival_gate
        waiting = self.running and gate is not None and gate.waiting_for_next
        ready = waiting and gate.ready and not self.stop_event.is_set() and not gate.next_requested
        last = self.active_index is not None and self.active_index == self.run_indices[-1] if self.running else False
        self.next_button.config(state="normal" if ready else "disabled", text="Finish" if last else "Next point")
        if waiting and self.active_index is not None and not self.stop_event.is_set():
            self.table.set(str(self.active_index), "state", "Await Next" if ready else "Arriving")
            self.status.config(text=(
                f"Point {self.active_index + 1} reached. Press {'Finish' if last else 'Next point'} when ready."
                if ready else "Robot left tolerance or tracking is incomplete. Waiting for arrival again; Next is blocked."
            ))

    def poll(self):
        self.after_id = None
        try:
            while True:
                job, kind, index, result = self.events.get_nowait()
                if job != self.job:
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
                    self.set_running(False)
                    for button in (self.load_button, self.calibration_button, self.run_button, self.run_selected_button):
                        button.config(state="normal")
                    self.stop_button.config(state="disabled")
                    self.units_combo.config(state="readonly")
                    self.extrapolation_check.config(state="normal")
                    self.arrival_gate = None
                    self.next_button.config(state="disabled", text="Next point")
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
        if self.pixels is not None and self.show_all:
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
        if self.setup is None:
            return
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
        if getattr(self, 'camera_popout', None) is not None:
            self.camera_popout.close()
        if getattr(self, 'recording', None) is not None:
            self.stop_recording()
        self.stop_event.set()
        self.job += 1
        if self.after_id is not None:
            self.parent.after_cancel(self.after_id)
            self.after_id = None
