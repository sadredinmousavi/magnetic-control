"""GUI005 calibration using GUI004's dark-contrast moving swarm ROI."""

import csv
from datetime import datetime
import json
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from PIL import Image, ImageTk

from gui004 import (
    OfflineDetectionGUI, calibrated_dwell_coordinates, circle_detection_mask,
    exclude_calibration_markers, match_compact_swarm_candidates,
    robot_center_for, workspace_detection_mask,
)
from robot_vision import RobotDetector
from axis_calibration import AXIS_TARGETS_M, AXIS_MOVE_TARGETS_M, AXIS_SAVED_TARGETS_M, AXIS_REFERENCE_NAMES, AXIS_DISPLAY_NAMES, RobotAxes, symmetric_axes, manual_line_axes
from point_control import hardware_case_name
from camera_popout import CameraPopout, show_view_dialog
from hover_hint import HoverHint


SETUP_DIR = Path(__file__).resolve().parent / "setups"
DEFAULT_ROBOT_ROI_RADIUS = 300
DEFAULT_PIN_ROI_RADIUS = 35


def default_setup_name():
    return datetime.now(ZoneInfo("Asia/Tehran")).strftime("%Y%m%d_%H%M%S.json")


class MovingROICalibration:
    """Frame-by-frame moving-ROI tracking for live camera previews."""

    def __init__(self, corners, width_cm, height_cm, circle, count, minimum_area, axis_samples=None,
                 contrast_threshold=10, strict_single=False, limit_to_pins=True, allow_uncalibrated=False, axis_capture_moves=None):
        self.metric_ready = len(corners) == 4
        if self.metric_ready or not allow_uncalibrated:
            calibrated_dwell_coordinates([], corners, width_cm, height_cm)
        if count < 1 or not np.isfinite(minimum_area) or minimum_area <= 0:
            raise ValueError("Robot count and minimum area must be positive.")
        circle = np.asarray(circle, dtype=float)
        if circle.shape != (3,) or not np.isfinite(circle).all() or circle[2] <= 0:
            raise ValueError("Place a circular ROI around the robots first.")
        self.corners = list(corners)
        self.width_cm, self.height_cm = width_cm, height_cm
        self.center = circle[:2].copy()
        self.radius = float(circle[2])
        self.count, self.minimum_area = count, minimum_area
        self.references = []
        self.strict_single = strict_single
        self.limit_to_pins = limit_to_pins and self.metric_ready
        self.workspace_mask = None
        self.workspace_circle = None  # Optional fixed display/processing workspace, in native pixels.
        self.excluded_mask = None  # Optional full-frame cargo mask for robot detection.
        self.selection_error = None
        self.detector = RobotDetector()
        self.axes = RobotAxes(corners, width_cm, height_cm, axis_samples, axis_capture_moves) if axis_samples else None
        self.contrast_threshold = float(contrast_threshold)
        if not np.isfinite(self.contrast_threshold) or not 1 <= self.contrast_threshold <= 255:
            raise ValueError("Dark contrast threshold must be between 1 and 255.")

    def coordinates(self, points):
        if not self.metric_ready:
            raise ValueError('Physical XY requires four calibrated pins.')
        if self.axes is not None:
            return self.axes.coordinates(points)
        return calibrated_dwell_coordinates(
            [{"center": point} for point in points], self.corners, self.width_cm, self.height_cm,
        )

    def detect_candidates(self, frame):
        # Keep native pixels, but blur/label only the ROI neighbourhood. The
        # halo preserves the sigma-5 background estimate at the ROI boundary.
        h, w = frame.shape[:2]
        halo = 32
        x0 = max(0, min(w, int(np.floor(self.center[0] - self.radius)) - halo))
        y0 = max(0, min(h, int(np.floor(self.center[1] - self.radius)) - halo))
        x1 = max(0, min(w, int(np.ceil(self.center[0] + self.radius)) + halo + 1))
        y1 = max(0, min(h, int(np.ceil(self.center[1] + self.radius)) + halo + 1))
        if self.workspace_circle is not None:
            cx, cy, radius = self.workspace_circle
            # Retain real-image halo for the blur; never blur a black display mask.
            x0 = max(x0, int(np.floor(cx - radius)) - halo)
            y0 = max(y0, int(np.floor(cy - radius)) - halo)
            x1 = min(x1, int(np.ceil(cx + radius)) + halo + 1)
            y1 = min(y1, int(np.ceil(cy + radius)) + halo + 1)
        if x1 <= x0 or y1 <= y0:
            return []
        crop = frame[y0:y1, x0:x1]
        offset = np.array([x0, y0])
        mask = circle_detection_mask(crop.shape[:2], (*(self.center - offset), self.radius))
        if self.workspace_circle is not None:
            cx, cy, radius = self.workspace_circle
            mask = cv2.bitwise_and(mask, circle_detection_mask(crop.shape[:2], (cx-x0, cy-y0, radius)))
        if self.limit_to_pins:
            # Rasterize once in full-frame coordinates so fractional corners
            # and slanted pin boundaries exactly match the original mask.
            if self.workspace_mask is None or self.workspace_mask.shape != frame.shape[:2]:
                self.workspace_mask = workspace_detection_mask(frame.shape[:2], self.corners)
            mask = cv2.bitwise_and(mask, self.workspace_mask[y0:y1, x0:x1])
        if self.excluded_mask is not None:
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(self.excluded_mask[y0:y1, x0:x1]))
        _, candidates = self.detector.process(
            crop, mode="Color blobs", color="Dark contrast",
            minimum_area=self.minimum_area, morphology_kernel_size=1,
            draw_annotations=False, detection_mask=mask,
            dark_contrast_threshold=self.contrast_threshold,
        )
        for item in candidates:
            item['center'] = (item['center'][0] + x0, item['center'][1] + y0)
            x, y, width, height = item['bbox']
            item['bbox'] = (x + x0, y + y0, width, height)
        return candidates

    def process(self, frame):
        candidates = self.detect_candidates(frame)
        candidates, _ = exclude_calibration_markers(candidates, self.corners, 10)
        self.selection_error = None
        if self.references:
            selected = match_compact_swarm_candidates(
                candidates, self.references, self.count, self.radius,
                predicted_center=self.center, swarm_radius=self.radius,
            )
        else:
            selected = [dict(item, robot_id=index + 1)
                        for index, item in enumerate(candidates[:self.count])]
        if self.strict_single and selected:
            # Reuse normal selection, but don't silently acquire/switch between
            # similarly strong blobs without a clear previous-position match.
            chosen = selected[0]
            for other in candidates:
                if other["center"] == chosen["center"]:
                    continue
                comparable = other["area"] >= .8 * chosen["area"]
                if not self.references:
                    ambiguous = comparable
                else:
                    previous = np.asarray(self.references[0])
                    chosen_distance = np.linalg.norm(np.asarray(chosen["center"]) - previous)
                    other_distance = np.linalg.norm(np.asarray(other["center"]) - previous)
                    ambiguous = (comparable and other_distance <= chosen_distance + 5) or (
                        other["area"] >= .5 * chosen["area"] and other_distance + 5 < chosen_distance)
                if ambiguous:
                    self.selection_error = "Robot identity is ambiguous between multiple strong candidates. Adjust the ROI or detection settings; capture has not been saved."
                    selected = []
                    break
        center = robot_center_for(selected, self.count)
        # Only a complete group moves the ROI or produces a valid group XY.
        # Missing robots leave a gap rather than shifting toward a partial group.
        if center is not None:
            self.center = np.asarray(center, dtype=float)
            self.references = [item["center"] for item in sorted(
                selected, key=lambda item: item["robot_id"])]
        xy = self.coordinates([item['center'] for item in selected]) if self.metric_ready else None
        for index, item in enumerate(selected):
            item['xy_cm'] = None if xy is None else tuple(xy[index])
        group_xy = None if center is None or not self.metric_ready else tuple(self.coordinates([center])[0])
        return selected, center, group_xy


class CalibrationTab:
    def __init__(self, parent, axis_move=None, axis_prepare=None, *, visible=lambda: True,
                 helmholtz_center_changed=None, helmholtz_edit_allowed=lambda: True):
        self.parent = parent
        self.visible = visible
        self.frame = None
        self.corners = []
        self.circle = None
        self.pin_search_circle = None
        self.helmholtz_circle = None
        self.helmholtz_center_changed = helmholtz_center_changed
        self.helmholtz_edit_allowed = helmholtz_edit_allowed
        self.helmholtz_radius_var = tk.StringVar(value='150')
        self.transform = None
        self.tracker = None
        self.auto_tracking = True
        self.auto_tracker_signature = None
        self.setup_revision = 0
        self.camera_source_key = 'Pi SSH'
        self.detections = []
        self.frame_number = 0
        self.render_revision = 0
        self.rows = []
        self.live_started = None
        self.detector = RobotDetector()
        self.axis_move = axis_move
        self.axis_prepare = axis_prepare
        self.axis_prepare_busy = False
        self.prepared_case = None
        self.prepare_job = 0
        self.prepare_events = queue.Queue()
        self.prepare_after = None
        self.axis_samples = {}
        self.axis_capture_moves = {}
        self.axis_raw_captures = {}
        self.axis_line_points = {}
        self.axis_method_name = 'Symmetric pairs'
        self.axis_method_var = tk.StringVar(value='Symmetric pairs')
        self.axis_pending = None
        self.axis_busy = False
        self.axis_events = queue.Queue()
        self.axis_job = 0
        self.axis_after = None
        self.frame_received_at = None
        self.axis_ready_at = None
        self.axis_tracker = None
        self.axis_detections = []
        self.axis_processed_at = None
        self.camera_popout = None
        self.command_running = False

        camera_view_bar = tk.Frame(parent, background='#edf4ff')
        camera_view_bar.pack(fill='x', padx=8, pady=(6, 2))
        tk.Label(camera_view_bar, text='Camera view', background='#edf4ff', foreground='#173c68',
                 font=('Segoe UI', 10, 'bold')).pack(side='left', padx=8)
        self.popout_button = tk.Button(camera_view_bar, text='Pop-out camera', command=self.open_camera_popout,
                                       background='#195da8', foreground='white', activebackground='#12477e',
                                       activeforeground='white', font=('Segoe UI', 10, 'bold'), padx=12)
        self.popout_button.pack(side='left', padx=8, pady=4)
        self.live_fps_var = tk.StringVar(value='5')
        tk.Label(camera_view_bar, text='Live processing FPS', background='#edf4ff', foreground='#173c68').pack(side='left')
        tk.Spinbox(camera_view_bar, from_=1, to=15, textvariable=self.live_fps_var, width=3).pack(side='left', padx=4)
        tk.Label(camera_view_bar, text='Opens the larger camera window; all controls remain here.',
                 background='#edf4ff', foreground='#173c68').pack(side='left', padx=4)
        self.step_labels = self.build_step_status(parent)
        bar = tk.Frame(parent)
        bar.pack(fill="x", padx=8, pady=6)
        for label, command in (
            ("Save setup", self.save_setup), ("Load setup", self.load_setup),
            ("Load last", self.load_last_setup),
            ("Export XY CSV", self.export_csv),
        ):
            tk.Button(bar, text=label, command=command).pack(side="left", padx=2)
        settings = tk.Frame(parent)
        settings.pack(fill="x", padx=8, pady=3)
        self.width_var = tk.StringVar(value="10")
        self.height_var = tk.StringVar(value="10")
        self.radius_var = tk.StringVar(value=str(DEFAULT_ROBOT_ROI_RADIUS))
        self.pin_radius_var = tk.StringVar(value=str(DEFAULT_PIN_ROI_RADIUS))
        self.manual_pin_var = tk.BooleanVar(value=False)
        self.count_var = tk.StringVar(value="1")
        self.area_var = tk.StringVar(value="1")
        self.contrast_var = tk.StringVar(value="10")
        self.show_mask_var = tk.BooleanVar(value=False)
        for label, variable in (("Width (cm)", self.width_var),
                                ("Height (cm)", self.height_var),
                                ("ROI radius (px)", self.radius_var),
                                ("Min area (px²)", self.area_var)):
            tk.Label(settings, text=label).pack(side="left", padx=(4, 2))
            tk.Entry(settings, textvariable=variable, width=6).pack(side="left")
        tk.Label(settings, text='Phase 2: one robot (fixed)').pack(side='left', padx=8)
        sensitivity = tk.Frame(parent)
        sensitivity.pack(fill="x", padx=8, pady=3)
        tk.Label(sensitivity, text="Dark contrast threshold").pack(side="left")
        contrast_entry = tk.Spinbox(sensitivity, from_=1, to=255, textvariable=self.contrast_var,
                                    width=5, command=self.render)
        contrast_entry.pack(side="left", padx=4)
        contrast_entry.bind("<KeyRelease>", lambda event: self.render())
        tk.Label(sensitivity, text="Lower detects fainter robots").pack(side="left", padx=4)
        tk.Checkbutton(sensitivity, text="Show detection mask", variable=self.show_mask_var,
                       command=self.render).pack(side="left", padx=4)
        self.detection_info = tk.Label(sensitivity, text="", anchor="w")
        self.detection_info.pack(side="left", padx=4)
        tools = tk.Frame(parent)
        tools.pack(fill="x", padx=8, pady=4)
        self.mode = tk.StringVar(value="ROI")
        for label, value in (("Find 4 green pins (small ROI)", "Corners"), ("Place robot ROI center", "ROI")):
            ttk.Radiobutton(tools, text=label, value=value, variable=self.mode,
                            command=self.clear_pin_roi).pack(side="left", padx=4)
        tk.Button(tools, text="Clear points", command=self.clear_points).pack(side="left", padx=4)
        tk.Button(tools, text='Clear ROI', command=self.clear_roi).pack(side='left', padx=4)
        coil_tools = tk.Frame(parent)
        coil_tools.pack(fill='x', padx=8, pady=2)
        ttk.Radiobutton(coil_tools, text='Align Helmholtz circle (click / drag)',
                        variable=self.mode, value='Helmholtz', command=self.clear_pin_roi).pack(side='left')
        tk.Label(coil_tools, text='Circle radius px').pack(side='left', padx=4)
        coil_radius = tk.Spinbox(coil_tools, from_=5, to=4000, width=6,
                                textvariable=self.helmholtz_radius_var, command=self.resize_helmholtz_circle)
        coil_radius.pack(side='left')
        coil_radius.bind('<Return>', lambda event: self.resize_helmholtz_circle())
        coil_radius.bind('<FocusOut>', lambda event: self.resize_helmholtz_circle())
        tk.Button(coil_tools, text='Clear coil alignment', command=self.clear_helmholtz_circle).pack(side='left', padx=4)
        self.helmholtz_status = tk.Label(coil_tools, text='Coil center: settings-file fallback', anchor='w')
        self.helmholtz_status.pack(side='left', padx=4)
        pin_tools = tk.Frame(parent)
        pin_tools.pack(fill="x", padx=8, pady=2)
        tk.Label(pin_tools, text="Pin search radius (px)").pack(side="left")
        tk.Spinbox(pin_tools, from_=3, to=300, textvariable=self.pin_radius_var, width=5).pack(side="left", padx=4)
        tk.Checkbutton(pin_tools, text="Manual pin placement", variable=self.manual_pin_var,
                       command=self.on_pin_method_changed).pack(side="left", padx=4)
        tk.Label(pin_tools, text="Auto: click the small ROI over a green pin. Manual: click the exact pin center.").pack(side="left", padx=4)
        axes_bar = tk.LabelFrame(parent, text="Workspace XY axes — robot captures or manual lines", padx=5, pady=4)
        axes_bar.pack(fill="x", padx=8, pady=4)
        axes_bar.columnconfigure(1, weight=1)
        self.control_case_var = tk.StringVar(value="")
        cases_dir = Path(__file__).resolve().parents[1] / "cases"
        cases = [path.parent.name for path in sorted(cases_dir.glob("case_*/case.py"))]
        tk.Label(axes_bar, text="Hardware case").grid(row=0, column=0)
        case_row = tk.Frame(axes_bar)
        case_row.grid(row=0, column=1, padx=4, sticky='ew')
        ttk.Combobox(case_row, textvariable=self.control_case_var, values=cases,
                     width=28, state="readonly").pack(side='left')
        self.prepare_button = tk.Button(case_row, text='Prepare axis moves', command=self.prepare_axis_moves)
        self.prepare_button.pack(side='left', padx=4)
        tk.Label(case_row, text='Method:').pack(side='left', padx=(8, 2))
        method = ttk.Combobox(case_row, textvariable=self.axis_method_var,
                              values=('Symmetric pairs', 'Center reference', 'Manual axis lines'), state='readonly', width=19)
        self.axis_method_selector = method
        method.pack(side='left')
        method.bind('<<ComboboxSelected>>', lambda event: self.change_axis_method())
        actions = tk.Frame(axes_bar)
        actions.grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 0))
        self.axis_buttons = []
        for name in AXIS_MOVE_TARGETS_M:
            button = tk.Button(actions, text=AXIS_DISPLAY_NAMES[name], width=6 if name == 'Origin' else 4,
                               command=lambda n=name: self.move_axis(n))
            button.pack(side="left", padx=(6 if name in ('+X', '+Y') else 1, 1))
            self.style_axis_move_button(button, name)
            self.axis_buttons.append(button)
        self.manual_lines_button = tk.Button(actions, text='Start / restart lines', command=self.start_manual_lines)
        self.manual_lines_button.pack(side='left', padx=3)
        self.undo_line_button = tk.Button(actions, text='Undo line point', command=self.undo_line_point)
        self.undo_line_button.pack(side='left', padx=3)
        self.axis_capture_button = tk.Button(actions, text="Capture", command=self.capture_axis, state="disabled")
        self.axis_capture_button.pack(side="left", padx=3)
        tk.Button(actions, text="Reset axes", command=self.reset_axes).pack(side="left", padx=3)
        tk.Label(axes_bar, anchor="w", justify="left", text=(
            "Choose the case matching your actual magnet count, ring radius, magnet sizes and magnetic properties. "
            "Only the case.py hardware model is used; no condition or sequence file is needed."
        ), wraplength=1000).grid(row=2, column=0, columnspan=4, sticky="ew", pady=(4, 0))
        self.axis_status = tk.Label(axes_bar, anchor="w", background='#edf4ff', foreground='#173c68', padx=6, pady=4, text=(
            "Ratio = 1. -X/-Y = -10 mm, X/Y = 10 mm, 2X/2Y = 20 mm from Center. The ROI follows one robot. "
            "Prepare axis moves once. Moves can be repeated in any order; Capture saves only the latest completed move. "
            "Symmetric pairs: capture -X, +X, -Y, +Y in any order; Center and +2X/+2Y are checks. Center reference: capture Center first, then one X and Y position."
        ), wraplength=1000)
        self.axis_status.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(4, 0))
        self.refresh_axis_method_controls()
        self.capture_hint = tk.Label(axes_bar, anchor='w', foreground='#855300', background='#fff4d6', padx=6, pady=3)
        self.capture_hint.grid(row=4, column=0, columnspan=4, sticky='ew')
        tk.Label(parent, anchor="w", text=(
            "Select four corner dots, enter their rectangle size, then place a circular ROI around all robots. "
            "Dark contrast detection only. Calibrate workspace axes using robot captures or manual pen lines."
        ), wraplength=1000).pack(fill="x", padx=10)
        self.status = tk.Label(parent, text="Waiting for a live camera frame.", anchor="w",
                               background='#edf4ff', foreground='#173c68', padx=6, pady=4)
        self.status.pack(fill="x", padx=10, pady=4)
        self.canvas = tk.Canvas(parent, background="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.canvas.bind("<Button-1>", self.click)
        self.canvas.bind('<B1-Motion>', self.drag_helmholtz_circle)
        self.canvas.bind("<Motion>", self.move_pin_roi)
        self.canvas.bind("<Leave>", self.clear_pin_roi)
        self.canvas.bind("<Configure>", lambda event: self.render())
        for variable in (self.width_var, self.height_var, self.control_case_var):
            variable.trace_add("write", self.on_metric_size_changed)

    def on_metric_size_changed(self, *args):
        if getattr(self, 'prepared_case', None) != self.control_case_var.get():
            self.prepared_case = None
        if self.axis_samples or getattr(self, 'axis_raw_captures', {}) or getattr(self, 'axis_line_points', {}) or self.axis_busy or self.axis_pending or getattr(self, "axis_tracker", None) is not None:
            self.reset_axes()
            self.axis_status.config(text="Rectangle size or control case changed. Recalibrate the robot axes.")

    def clear_tracking(self):
        self.tracker = None
        self.detections = []
        self.rows = []
        self.live_started = None
        self.auto_tracker_signature = None

    def ensure_roi_tracking(self):
        if self.circle is None:
            self.tracker = self.axis_tracker = None
            self.detections = []
            return
        samples = self.axis_samples if len(self.axis_samples) == 3 else None
        signature = json.dumps([self.corners, self.width_var.get(), self.height_var.get(), samples, getattr(self, "axis_capture_moves", {})], sort_keys=True)
        if self.tracker is None or signature != self.auto_tracker_signature:
            circle = self.circle if self.tracker is None else (*self.tracker.center, self.tracker.radius)
            self.tracker = MovingROICalibration(self.corners, float(self.width_var.get()), float(self.height_var.get()),
                circle, 1, float(self.area_var.get()), samples, strict_single=True,
                axis_capture_moves=getattr(self, "axis_capture_moves", {}),
                contrast_threshold=self.read_contrast_threshold(), allow_uncalibrated=True)
            self.axis_tracker = self.tracker
            self.auto_tracker_signature = signature
            if self.live_started is None:
                self.live_started = time.monotonic()
        self.tracker.minimum_area = float(self.area_var.get())
        self.axis_tracker = self.tracker
        self.tracker.radius = float(self.radius_var.get())
        if not np.isfinite([self.tracker.minimum_area, self.tracker.radius]).all() or self.tracker.minimum_area <= 0 or self.tracker.radius <= 0:
            raise ValueError('Minimum area and ROI radius must be positive and finite.')

    def clear_roi(self):
        self.circle = None
        self.tracker = self.axis_tracker = None
        self.detections, self.axis_detections = [], []
        self.auto_tracker_signature = None
        self.axis_pending = self.axis_ready_at = None
        self.axis_processed_at = self.axis_processed_settings = None
        self.mask_preview_key = self.mask_preview = None
        if hasattr(self, 'axis_capture_button'):
            self.axis_capture_button.config(state='disabled')
        self.status.config(text='ROI cleared. Tracking stopped; place a new ROI to restart.')
        if hasattr(self, 'axis_status'):
            self.axis_status.config(text='ROI cleared; pending capture cancelled. Saved axis samples are unchanged. '
                                    'Any motor move already sent is not undone.')
        self.render()

    def style_axis_move_button(self, button, name):
        reference = AXIS_REFERENCE_NAMES[name]
        background, foreground = (('#edf4ff', '#174f91') if reference == '+X' else
                                  ('#e4f5e9', '#176637') if reference == '+Y' else ('#f0f1f3', '#30343a'))
        button.config(background=background, foreground=foreground, padx=3, pady=2,
                      width=6 if name == 'Origin' else 4)
        target = AXIS_MOVE_TARGETS_M[name]
        HoverHint(button, f'Move to ({target[0]*1000:g}, {target[1]*1000:g}) mm from Center, ratio 1. '
                  'Capture saves this position. Symmetric pairs keeps opposite captures separately; Center and +2 moves check the fit.')

    def read_contrast_threshold(self):
        variable = getattr(self, "contrast_var", None)
        value = float(variable.get()) if variable is not None else 10.0
        if not np.isfinite(value) or not 1 <= value <= 255:
            raise ValueError("Dark contrast threshold must be between 1 and 255.")
        return value

    def reset_axes(self):
        self.stop_tracking()
        self.axis_job += 1
        self.axis_busy = False
        self.axis_pending = None
        self.axis_ready_at = None
        self.axis_samples = {}
        self.axis_capture_moves = {}
        self.axis_raw_captures = {}
        self.axis_line_points = {}
        self.axis_tracker = None
        self.axis_detections = []
        self.axis_processed_at = None
        for button in self.axis_buttons:
            button.config(state="normal")
        if hasattr(self, "axis_capture_button"):
            self.axis_capture_button.config(state="disabled")
        self.axis_status.config(text=self.axis_instructions())
        if self.manual_lines_method():
            self.mode.set('AxisLines')
        elif getattr(self, 'mode', None) is not None and self.mode.get() == 'AxisLines':
            self.mode.set('ROI')
        self.refresh_axis_method_controls()
        self.render()

    def symmetric_method(self):
        variable = getattr(self, 'axis_method_var', None)
        return variable is not None and variable.get() == 'Symmetric pairs'

    def manual_lines_method(self):
        variable = getattr(self, 'axis_method_var', None)
        return variable is not None and variable.get() == 'Manual axis lines'

    def refresh_axis_method_controls(self):
        manual = self.manual_lines_method()
        locked = (getattr(self, 'command_running', False) or getattr(self, 'axis_busy', False)
                  or getattr(self, 'axis_prepare_busy', False))
        for name in ('axis_method_selector',):
            if hasattr(self, name):
                getattr(self, name).config(state='disabled' if locked else 'readonly')
        for button in getattr(self, 'axis_buttons', []):
            button.config(state='disabled' if manual or locked else 'normal')
        if hasattr(self, 'prepare_button'):
            self.prepare_button.config(state='disabled' if manual or locked else 'normal')
        for name in ('manual_lines_button', 'undo_line_button'):
            if hasattr(self, name):
                getattr(self, name).config(state='normal' if manual and not locked else 'disabled')
        if manual and hasattr(self, 'axis_capture_button'):
            self.axis_capture_button.config(state='disabled')

    def change_axis_method(self):
        if self.helmholtz_locked():
            self.axis_method_var.set(getattr(self, 'axis_method_name', 'Symmetric pairs'))
            self.axis_status.config(text='Finish the current operation before changing axis calibration.')
            return
        self.axis_method_name = self.axis_method_var.get()
        self.reset_axes()

    def start_manual_lines(self):
        if not self.manual_lines_method() or self.helmholtz_locked():
            return
        if self.frame is None or len(self.corners) != 4:
            self.axis_status.config(text='Select the four workspace pins and physical dimensions first.')
            return
        self.reset_axes()
        self.mode.set('AxisLines')
        self.status.config(text='Place the pen along X. Click -X, then +X; reposition along Y and click -Y, then +Y. Keep camera and workspace fixed.')

    def undo_line_point(self):
        if not self.manual_lines_method() or self.helmholtz_locked():
            return
        points = dict(getattr(self, 'axis_line_points', {}))
        for name in reversed(('-X', '+X', '-Y', '+Y')):
            if name in points:
                del points[name]
                break
        self.axis_line_points = points
        self.axis_samples, self.axis_capture_moves = {}, {}
        self.clear_tracking()
        self.mode.set('AxisLines')
        self.setup_revision = getattr(self, 'setup_revision', 0)+1
        self.axis_status.config(text=self.axis_instructions())
        self.render()

    def record_line_point(self, point):
        if not self.manual_lines_method() or self.helmholtz_locked():
            return
        if self.frame is None or len(self.corners) != 4:
            self.axis_status.config(text='Select the four workspace pins and physical dimensions first.')
            return
        points = dict(getattr(self, 'axis_line_points', {}))
        if len(points) == 4:
            self.axis_status.config(text=self.axis_instructions())
            return
        try:
            width, height = float(self.width_var.get()), float(self.height_var.get())
            if not np.isfinite([width, height]).all() or min(width, height) <= 0:
                raise ValueError('Workspace width and height must be finite and positive.')
            name = ('-X', '+X', '-Y', '+Y')[len(points)]
            point = np.asarray(point, float)
            h, w = self.frame.shape[:2]
            if point.shape != (2,) or not np.isfinite(point).all() or np.any(point < 0) or np.any(point >= [w, h]):
                raise ValueError('Click a point inside the camera image.')
            if name in ('+X', '+Y') and np.linalg.norm(point-np.asarray(points['-'+name[-1]])) < 2:
                raise ValueError('Select line endpoints farther apart (at least 2 pixels).')
            points[name] = point.tolist()
            samples, report = {}, ''
            if len(points) == 4:
                samples, report = manual_line_axes(self.corners, width, height, points)
                origin = np.asarray(samples['Origin'])
                if np.any(origin < 0) or np.any(origin >= [w, h]):
                    raise ValueError('The axis intersection is outside the image. Check the pen lines.')
            self.axis_line_points = points
            self.axis_samples = samples
            self.axis_capture_moves = {key: key for key in samples}
            self.clear_tracking()
            self.setup_revision = getattr(self, 'setup_revision', 0)+1
            self.axis_status.config(text=('Manual axis lines complete. '+report+' Save setup to retain the axes.'
                                          if samples else self.axis_instructions()))
            self.render()
        except (ValueError, TypeError) as error:
            # Keep earlier clicks, so a rejected last point can be clicked again.
            self.axis_status.config(text=f'Point not saved: {error}')

    def axis_instructions(self):
        if self.manual_lines_method():
            count = len(getattr(self, 'axis_line_points', {}))
            if count == 4:
                return 'Manual axis lines complete. Save setup; Undo line point adjusts +Y, or restart both lines.'
            name = ('-X', '+X', '-Y', '+Y')[count]
            return f'Manual axis lines: click {name} on the pen. Order: -X, +X, -Y, +Y. No robot or motor move required.'
        if self.symmetric_method():
            raw = getattr(self, 'axis_raw_captures', {})
            missing = [name for name in ('-X', '+X', '-Y', '+Y') if name not in raw]
            return ('Symmetric pairs: capture ' + ', '.join(missing) +
                    '. Center and +2X/+2Y are optional checks; ratio is 1.')
        return 'Center reference: capture Center first, then one X and one Y position. Ratio is 1.'

    def record_symmetric_capture(self, center):
        captures = dict(getattr(self, 'axis_raw_captures', {}))
        captures[self.axis_pending] = list(center)
        samples, report = {}, self.axis_instructions()
        if all(name in captures for name in ('-X', '+X', '-Y', '+Y')):
            samples, report = symmetric_axes(self.corners, float(self.width_var.get()),
                                              float(self.height_var.get()), captures)
        self.axis_raw_captures = captures
        self.axis_samples = samples
        self.axis_capture_moves = {name: name for name in samples}
        self.axis_pending = self.axis_ready_at = None
        self.axis_capture_button.config(state='disabled')
        for button in self.axis_buttons:
            button.config(state='normal')
        self.axis_status.config(text=('Symmetric calibration complete. ' + report if samples
                                     else self.axis_instructions()))
        self.circle = (*self.axis_tracker.center, self.axis_tracker.radius)
        self.render()

    def prepare_axis_moves(self):
        if self.manual_lines_method():
            self.axis_status.config(text=self.axis_instructions())
            return
        if self.axis_busy or self.axis_prepare_busy or self.command_running:
            self.axis_status.config(text='Wait for the current operation to finish before preparing axes.')
            return
        case = self.control_case_var.get()
        if not case or self.axis_prepare is None:
            show_view_dialog(self, messagebox.showerror, 'Prepare axes', 'Select a hardware case; axis preparation must be available.')
            return
        self.prepared_case = None
        self.axis_prepare_busy = True
        self.prepare_job += 1
        job = self.prepare_job
        self.prepare_button.config(state='disabled')
        for button in self.axis_buttons:
            button.config(state='disabled')
        self.axis_status.config(text='Preparing Center and X/Y at -10, +10 and +20 mm in the background. No motors are moved.')

        def worker():
            try:
                self.axis_prepare(case)
                self.prepare_events.put((job, case, None))
            except Exception as error:
                self.prepare_events.put((job, case, str(error)))

        threading.Thread(target=worker, daemon=True).start()
        self.prepare_after = self.parent.after(50, self.poll_axis_preparation)

    def poll_axis_preparation(self):
        self.prepare_after = None
        try:
            job, case, error = self.prepare_events.get_nowait()
        except queue.Empty:
            if self.axis_prepare_busy:
                self.prepare_after = self.parent.after(50, self.poll_axis_preparation)
            return
        if job != self.prepare_job:
            return
        self.axis_prepare_busy = False
        self.prepare_button.config(state='normal')
        for button in self.axis_buttons:
            button.config(state='normal')
        if error:
            self.axis_status.config(text=f'Axis preparation failed: {error}')
        elif case != self.control_case_var.get():
            self.axis_status.config(text='Case changed during preparation. Prepare the selected case again.')
        else:
            self.prepared_case = case
            self.axis_status.config(text='Axis poses ready. Moves send prepared angles immediately; no timed settle wait. Capture manually when the robot has settled.')
        self.render()

    def move_axis(self, name):
        if self.manual_lines_method():
            self.axis_status.config(text=self.axis_instructions())
            return
        if self.axis_busy or getattr(self, 'axis_prepare_busy', False) or getattr(self, 'command_running', False):
            return
        try:
            if getattr(self, 'axis_prepare', None) is not None and self.prepared_case != self.control_case_var.get():
                raise ValueError('Press Prepare axis moves beside the selected hardware case first.')
            if not self.control_case_var.get():
                raise ValueError("Select the control case matching your hardware geometry first.")
            if self.axis_move is None:
                raise ValueError("Controller movement is unavailable.")
        except ValueError as error:
            show_view_dialog(self, messagebox.showerror, "Robot axes", str(error))
            return
        # Vision is optional for testing a prepared motor pose, mandatory only
        # when saving a calibration capture. Acquire it opportunistically.
        try:
            vision_ready = self.frame is not None and self.circle is not None and len(self.corners) == 4 and int(self.count_var.get()) == 1
            if vision_ready and self.axis_tracker is None:
                circle = self.circle if self.tracker is None else (*self.tracker.center, self.tracker.radius)
                self.axis_tracker = MovingROICalibration(
                    self.corners, float(self.width_var.get()), float(self.height_var.get()),
                    (*circle[:2], float(self.radius_var.get())), 1, float(self.area_var.get()),
                    contrast_threshold=self.read_contrast_threshold(), strict_single=True,
                )
            if vision_ready:
                self.update_axis_tracking()
            else:
                self.axis_tracker = None
                self.axis_detections = []
        except (ValueError, TypeError, cv2.error):
            self.axis_tracker = None
            self.axis_detections = []
        if not getattr(self, 'auto_tracking', False):
            self.stop_tracking()
            self.clear_tracking()
        self.axis_pending = None
        self.axis_busy = True
        self.axis_capture_button.config(state="disabled")
        self.axis_job += 1
        job = self.axis_job
        case = self.control_case_var.get()
        for button in self.axis_buttons:
            button.config(state="disabled")
        target = AXIS_MOVE_TARGETS_M[name]
        self.axis_status.config(text=f"Sending {AXIS_DISPLAY_NAMES[name]} {target} m with ratio 1. Pins/ROI/camera are not required for moving. Capture requires valid live calibration; any previous pending move is replaced.")

        def worker():
            try:
                self.axis_move(target, case, lambda: self.axis_job == job)
                self.axis_events.put((job, name, None))
            except Exception as error:
                self.axis_events.put((job, name, str(error)))

        threading.Thread(target=worker, daemon=True).start()
        if self.axis_after is None:
            self.axis_after = self.parent.after(50, self.poll_axis_move)

    def poll_axis_move(self):
        self.axis_after = None
        try:
            while True:
                job, name, error = self.axis_events.get_nowait()
                if job != self.axis_job:
                    continue
                self.axis_busy = False
                for button in self.axis_buttons:
                    button.config(state="normal")
                if error:
                    self.axis_status.config(text=f"Movement failed: {error}")
                    show_view_dialog(self, messagebox.showerror, "Robot axes", error)
                elif self.circle is not None:
                    self.axis_pending = name
                    self.axis_ready_at = time.monotonic()
                    self.axis_capture_button.config(state="normal")
                    self.axis_status.config(text=f"{AXIS_DISPLAY_NAMES[name]} command sent. Confirm visually that the robot has settled, then Capture to save this position. There is no automatic settle check or delay; moving again replaces the pending capture.")
                else:
                    self.axis_pending = self.axis_ready_at = None
                    self.axis_capture_button.config(state='disabled')
                    self.axis_status.config(text='Motor command finished without an ROI. Tracking remains stopped. '
                                                 'Place a new ROI, then move and capture to calibrate.')
        except queue.Empty:
            pass
        if self.axis_busy:
            self.axis_after = self.parent.after(50, self.poll_axis_move)

    def capture_axis(self):
        """User confirms settling; ambiguous or stale observations are not saved."""
        try:
            self.record_axis_from_roi()
        except (ValueError, TypeError) as error:
            self.axis_status.config(text=f"Capture not saved: {error}")
            show_view_dialog(self, messagebox.showerror, "Robot axes — capture not saved", str(error))

    def record_axis_from_roi(self):
        if self.axis_busy or not self.axis_pending:
            raise ValueError("Move to a calibration position first and wait for the command to finish.")
        received = self.frame_received_at
        if self.frame is None or received is None or time.monotonic() - received > 2:
            raise ValueError("No fresh live frame. Check the camera connection and try again.")
        if self.axis_ready_at is None or received <= self.axis_ready_at:
            raise ValueError("Wait for a new camera frame after the move, then capture again.")
        if int(self.count_var.get()) != 1:
            raise ValueError("Use exactly one robot and set Robots to 1.")
        if self.circle is None or len(self.corners) != 4:
            raise ValueError("Select four corners and place the ROI first.")
        area, radius = float(self.area_var.get()), float(self.radius_var.get())
        if not np.isfinite(area) or area <= 0 or not np.isfinite(radius) or radius <= 0:
            raise ValueError("Minimum area and ROI radius must be positive finite numbers.")
        if getattr(self, "axis_tracker", None) is None:
            self.axis_tracker = MovingROICalibration(
                self.corners, float(self.width_var.get()), float(self.height_var.get()),
                (*self.circle[:2], radius), 1, area,
                contrast_threshold=self.read_contrast_threshold(), strict_single=True,
            )
        settings = (area, radius, self.read_contrast_threshold())
        if (getattr(self, "axis_processed_at", None) != received
                or getattr(self, "axis_processed_settings", None) != settings):
            self.update_axis_tracking()
        if not self.axis_detections:
            raise ValueError(self.axis_tracker.selection_error or "No robot detected in the following ROI. Check the camera, ROI and Show detection mask; capture has not been saved.")
        center = self.axis_detections[0]["center"]
        if self.symmetric_method():
            self.record_symmetric_capture(center)
            return
        if self.axis_pending != "Origin" and "Origin" not in self.axis_samples:
            raise ValueError("Capture Center first.")
        if self.axis_pending != "Origin" and np.linalg.norm(np.asarray(center) - self.axis_samples["Origin"]) < 2:
            raise ValueError("Robot has not moved enough from Center (less than 2 pixels). Check the movement and wait for it to settle; nothing was saved.")
        # Only an explicit successful Center capture resets dependent X/Y samples.
        reference = AXIS_REFERENCE_NAMES[self.axis_pending]
        samples = dict({} if self.axis_pending == 'Origin' else self.axis_samples,
                       **{reference: list(center)})
        capture_moves = {key: getattr(self, 'axis_capture_moves', {}).get(key, key) for key in samples}
        capture_moves[reference] = self.axis_pending
        if len(samples) == 3:
            axes = RobotAxes(self.corners, float(self.width_var.get()), float(self.height_var.get()), samples, capture_moves)
            self.axis_status.config(text=(
                f"Robot axes calibrated. Observed X move: {axes.lengths_cm[0] * 10:.2f} mm; "
                f"Y move: {axes.lengths_cm[1] * 10:.2f} mm. References: {capture_moves['+X']}, {capture_moves['+Y']}. Save setup to retain this calibration."
            ))
        else:
            self.axis_status.config(text=f"{AXIS_DISPLAY_NAMES[self.axis_pending]} captured at ({center[0]:.1f}, {center[1]:.1f}) px. {len(samples)}/3 references recorded. Move to the next position.")
        self.axis_samples = samples
        self.axis_capture_moves = capture_moves
        self.axis_pending = None
        self.axis_ready_at = None
        if hasattr(self, "axis_capture_button"):
            self.axis_capture_button.config(state="disabled")
        for button in self.axis_buttons:
            button.config(state="normal")
        self.circle = (*self.axis_tracker.center, self.axis_tracker.radius)
        self.render()

    def update_axis_tracking(self):
        """Same moving-ROI detector as normal tracking, without exporting rows."""
        self.axis_tracker.contrast_threshold = self.read_contrast_threshold()
        self.axis_tracker.minimum_area = float(self.area_var.get())
        self.axis_tracker.radius = float(self.radius_var.get())
        if (not np.isfinite(self.axis_tracker.minimum_area) or self.axis_tracker.minimum_area <= 0
                or not np.isfinite(self.axis_tracker.radius) or self.axis_tracker.radius <= 0):
            raise ValueError("Minimum area and ROI radius must be positive finite numbers.")
        settings = (self.axis_tracker.minimum_area, self.axis_tracker.radius, self.axis_tracker.contrast_threshold)
        if self.tracker is self.axis_tracker and getattr(self, 'axis_processed_at', None) == self.frame_received_at and getattr(self, 'axis_processed_settings', None) == settings:
            return
        self.axis_detections, center, _ = self.axis_tracker.process(self.frame)
        self.axis_processed_at = self.frame_received_at
        self.axis_processed_settings = (self.axis_tracker.minimum_area, self.axis_tracker.radius,
                                        self.axis_tracker.contrast_threshold)
        self.detections = self.axis_detections
        if center is not None:
            self.circle = (*center, self.axis_tracker.radius)
        self.status.config(text=(
            "Axis tracking: robot found; ROI follows it. Confirm settling before Capture."
            if center is not None else self.axis_tracker.selection_error or
            "Axis tracking: robot lost; ROI held at the last position. Capture is blocked until it is found again."
        ))

    def accept_live_frame(self, frame):
        self.accept_frame(frame)

    def accept_frame(self, frame):
        if self.frame is not None and self.frame.shape[:2] != frame.shape[:2]:
            self.corners, self.circle = [], None
            self.helmholtz_circle = None
            self.pin_search_circle = None
            self.transform = None
            self.canvas.camera_transform = None
            popup = getattr(self, 'camera_popout', None)
            if popup is not None and popup.is_open:
                popup.canvas.camera_transform = None
            self.frame = frame
            self.reset_axes()
            self.clear_tracking()
            self.status.config(text="Frame size changed. Select four points and place the ROI again.")
        self.frame = frame
        self.frame_received_at = time.monotonic()
        popup = getattr(self, 'camera_popout', None)
        axis_needed = (getattr(self, 'axis_busy', False) or getattr(self, 'axis_pending', None)
                       or getattr(self, 'visible', lambda: True)()
                       or (popup is not None and popup.is_open))
        if not axis_needed:
            return
        if getattr(self, 'auto_tracking', False):
            try:
                self.ensure_roi_tracking()
            except (ValueError, TypeError) as error:
                self.tracker = self.axis_tracker = None
                self.detections = []
                self.status.config(text=f'Tracking waiting: {error}')
        if getattr(self, "axis_tracker", None) is not None and self.tracker is None and axis_needed:
            try:
                self.update_axis_tracking()
            except (ValueError, TypeError) as error:
                self.axis_detections = []
                self.axis_processed_at = None
                self.status.config(text=f"Axis tracking unavailable: {error}")
        if self.tracker is not None:
            try:
                self.tracker.contrast_threshold = self.read_contrast_threshold()
                self.detections, center, xy = self.tracker.process(frame)
                if getattr(self, 'auto_tracking', False):
                    self.axis_detections = self.detections
                    self.axis_processed_at = self.frame_received_at
                    self.axis_processed_settings = (self.tracker.minimum_area, self.tracker.radius, self.tracker.contrast_threshold)
                    if center is not None:
                        self.circle = (*self.tracker.center, self.tracker.radius)
                elapsed = time.monotonic() - self.live_started
                valid = center is not None
                for item in self.detections or [None]:
                    self.rows.append([
                        self.frame_number, elapsed, int(valid),
                        "" if item is None else item["robot_id"],
                        "" if item is None else item["center"][0],
                        "" if item is None else item["center"][1],
                        "" if item is None or item['xy_cm'] is None else item["xy_cm"][0],
                        "" if item is None or item['xy_cm'] is None else item["xy_cm"][1],
                        "" if xy is None else xy[0], "" if xy is None else xy[1],
                        *self.tracker.center,
                    ])
                detail = "Group missing: ROI held at last valid center."
                if valid and xy is not None:
                    detail = f"Group XY = ({xy[0]:.3f}, {xy[1]:.3f}) cm"
                elif valid:
                    detail = f'Robot pixel position {center}; physical XY unavailable until pins are calibrated.'
                self.status.config(text=f"Frame {self.frame_number}: {len(self.detections)}/{self.tracker.count} robots. {detail}")
            except Exception as error:
                self.stop_tracking()
                self.status.config(text=f"Tracking stopped: {error}")
        self.render()
        self.frame_number += 1

    def open_camera_popout(self):
        if self.camera_popout is not None and self.camera_popout.is_open:
            self.camera_popout.show()
            return
        self.camera_popout = CameraPopout(
            self.parent, 'GUI005 — Calibration camera', self.render,
            actions=(('Prepare', self.prepare_axis_moves, self.prepare_button),
                     *((AXIS_DISPLAY_NAMES[name], lambda n=name: self.move_axis(n), button)
                       for name, button in zip(AXIS_MOVE_TARGETS_M, self.axis_buttons)),
                     ('Capture', self.capture_axis, self.axis_capture_button),
                     ('Reset axes', self.reset_axes, None),
                     ('Clear pins', self.clear_points, None),
                     ('Clear ROI', self.clear_roi, None)),
            toolbar_builder=self.build_popout_tools, mirror_action_text=False,
            controls_enabled=lambda: not self.command_running,
            status_builder=self.build_step_status, status_refresh=self.refresh_step_status,
            render_version=lambda: self.render_revision,
            on_close=self.render,
            status=lambda: f"{self.capture_explanation()}  |  {self.axis_status.cget('text')}  |  {self.status.cget('text')}",
        )
        canvas = self.camera_popout.canvas
        for (button, _), name in zip(self.camera_popout.actions[1:], AXIS_MOVE_TARGETS_M):
            self.style_axis_move_button(button, name)
        canvas.bind('<Button-1>', lambda event: self.click(event, transform=getattr(canvas, 'camera_transform', None)))
        canvas.bind('<B1-Motion>', lambda event: self.drag_helmholtz_circle(event, transform=getattr(canvas, 'camera_transform', None)))
        canvas.bind('<Motion>', lambda event: self.move_pin_roi(event, transform=getattr(canvas, 'camera_transform', None)))
        canvas.bind('<Leave>', self.clear_pin_roi)
        # Keep the single-row controls reachable even when the view is resized.
        self.camera_popout.window.update_idletasks()
        toolbar_width = self.camera_popout.extra_controls.master.winfo_reqwidth()
        self.camera_popout.window.minsize(max(480, toolbar_width + 10), 320)
        self.render()

    def build_popout_tools(self, bar):
        # Shared variables keep both pages synchronized without extra setup dialogs.
        for label, mode in (('4 pins', 'Corners'), ('ROI', 'ROI'), ('Coil', 'Helmholtz')):
            tk.Radiobutton(bar, text=label, variable=self.mode, value=mode,
                           command=self.clear_pin_roi).pack(side='left', padx=1)
        tk.Button(bar, text='Axis lines', command=self.start_manual_lines).pack(side='left', padx=2)
        tk.Button(bar, text='Undo line', command=self.undo_line_point).pack(side='left', padx=2)
        tk.Checkbutton(bar, text='Manual', variable=self.manual_pin_var,
                       command=self.on_pin_method_changed).pack(side='left', padx=1)
        for label, variable in (('Pin r', self.pin_radius_var), ('ROI r', self.radius_var)):
            tk.Label(bar, text=label).pack(side='left', padx=(3, 1))
            tk.Entry(bar, textvariable=variable, width=4).pack(side='left')
        tk.Label(bar, text='Coil r').pack(side='left')
        radius = tk.Entry(bar, textvariable=self.helmholtz_radius_var, width=4)
        radius.pack(side='left')
        radius.bind('<Return>', lambda event: self.resize_helmholtz_circle())
        radius.bind('<FocusOut>', lambda event: self.resize_helmholtz_circle())
        tk.Label(bar, text='FPS').pack(side='left', padx=(3, 1))
        tk.Spinbox(bar, from_=1, to=15, textvariable=self.live_fps_var, width=3).pack(side='left')

    def build_step_status(self, parent):
        row = tk.Frame(parent)
        row.pack(fill='x', padx=8, pady=3)
        labels = []
        for title in ('1. 4 pins', '2. ROI and robot', '3. Axis'):
            label = tk.Label(row, text=title, anchor='w', padx=8, pady=4, font=('Segoe UI', 9))
            label.pack(side='left', fill='x', expand=True, padx=2)
            labels.append(label)
        return labels

    def calibration_progress(self):
        pins = len(self.corners) == 4
        try:
            count = int(self.count_var.get())
        except (ValueError, TypeError):
            count = 0
        robot = self.circle is not None and len(self.detections) == 1
        axes = len(self.axis_samples) == 3
        manual = self.manual_lines_method()
        line_count = len(getattr(self, 'axis_line_points', {}))
        return (
            ('1. 4 pins', 'Done' if pins else f'{len(self.corners)}/4 — select pins', pins, not pins),
            ('2. ROI and robot', 'Not needed for manual axes' if manual else 'Detected' if robot else 'Place ROI / detect robot', manual or robot, not manual and self.circle is not None and not robot),
            ('3. Axis', 'Done' if axes else (f'{line_count}/4 clicks' if manual else f'{sum(n in getattr(self, "axis_raw_captures", {}) for n in ("-X", "+X", "-Y", "+Y"))}/4 captures' if self.symmetric_method() else f'{len(self.axis_samples)}/3 captures'), axes, pins and (manual or self.circle is not None) and not axes),
        )

    def refresh_step_status(self, labels):
        for label, (title, detail, done, active) in zip(labels, self.calibration_progress()):
            background, foreground = (('#e4f5e9', '#176637') if done else
                                      ('#e4efff', '#174f91') if active else ('#f0f1f3', '#626772'))
            label.config(text=f'{title}  —  {detail}', background=background, foreground=foreground)

    def capture_explanation(self):
        if getattr(self, 'command_running', False):
            return 'Capture locked: stop the Phase 3 run before changing calibration.'
        if self.axis_busy:
            return 'Capture disabled: the motor command is still being sent.'
        if getattr(self, 'axis_prepare_busy', False):
            return 'Preparing axis poses: no motors are moved. Wait until preparation finishes.'
        if self.manual_lines_method():
            return self.axis_instructions()
        if self.axis_pending:
            return f'Capture ready for {AXIS_DISPLAY_NAMES[self.axis_pending]}: confirm the robot is settled, then press Capture.'
        if len(self.axis_samples) == 3:
            return 'Axes complete. Save setup; Capture becomes available after another axis move.'
        if self.symmetric_method():
            return self.axis_instructions()
        next_name = 'Center' if 'Origin' not in self.axis_samples else '+X' if '+X' not in self.axis_samples else '+Y'
        return f'Capture disabled: press Move to {next_name}, wait for the command to finish, then capture the settled robot.'

    def set_command_running(self, running):
        self.command_running = running
        self.refresh_axis_method_controls()
        if self.camera_popout is not None and self.camera_popout.is_open:
            self.camera_popout.refresh()

    def helmholtz_locked(self):
        return (getattr(self, 'command_running', False) or getattr(self, 'axis_busy', False)
                or getattr(self, 'axis_prepare_busy', False)
                or not getattr(self, 'helmholtz_edit_allowed', lambda: True)())

    def helmholtz_center_m(self):
        circle = getattr(self, 'helmholtz_circle', None)
        if circle is None or len(self.corners) != 4:
            return None
        samples = getattr(self, 'axis_samples', {})
        if samples and len(samples) != 3:
            return None
        if samples:
            axes = RobotAxes(self.corners, float(self.width_var.get()), float(self.height_var.get()),
                             samples, getattr(self, 'axis_capture_moves', {}))
            xy_cm = axes.coordinates([circle[:2]])[0]
        else:
            xy_cm = calibrated_dwell_coordinates([{'center': circle[:2]}], self.corners,
                                                  float(self.width_var.get()), float(self.height_var.get()))[0]
        return tuple(np.asarray(xy_cm, float)*.01)

    def refresh_helmholtz_center(self):
        if not hasattr(self, 'helmholtz_status'):
            return
        try:
            center = self.helmholtz_center_m()
            self.helmholtz_status.config(text=(f'Coil center XY: {center[0]*1000:.2f}, {center[1]*1000:.2f} mm'
                if center is not None else ('Coil center: settings-file fallback' if self.helmholtz_circle is None
                                            else 'Coil center: complete pins / axes')))
        except (ValueError, TypeError):
            center = None
            self.helmholtz_status.config(text='Coil center: complete valid XY calibration')
        callback = getattr(self, 'helmholtz_center_changed', None)
        if callback is not None:
            callback(center)

    def resize_helmholtz_circle(self):
        if self.helmholtz_locked():
            return
        try:
            radius = float(self.helmholtz_radius_var.get())
            if not np.isfinite(radius) or radius <= 0:
                raise ValueError('Coil circle radius must be positive.')
            if self.helmholtz_circle is not None:
                self.helmholtz_circle = (*self.helmholtz_circle[:2], radius)
                self.setup_revision += 1
            self.render()
        except ValueError as error:
            self.status.config(text=str(error))

    def clear_helmholtz_circle(self):
        if self.helmholtz_locked():
            return
        self.helmholtz_circle = None
        self.setup_revision += 1
        self.render()

    def drag_helmholtz_circle(self, event, *, transform=None):
        if self.mode.get() == 'Helmholtz':
            self.click(event, transform=transform)

    def click(self, event, *, transform=None):
        popup = getattr(self, 'camera_popout', None)
        if transform is None and popup is not None and popup.is_open:
            return
        if getattr(self, 'command_running', False):
            return
        transform = self.transform if transform is None else transform
        if self.frame is None or transform is None:
            return
        ox, oy, scale = transform
        point = (round((event.x - ox) / scale), round((event.y - oy) / scale))
        h, w = self.frame.shape[:2]
        if not (0 <= point[0] < w and 0 <= point[1] < h):
            return
        if self.axis_busy:
            self.axis_status.config(text="Wait for the move to finish, then use Capture settled robot.")
            return
        if self.mode.get() == 'AxisLines':
            self.record_line_point(point)
            return
        if self.mode.get() == 'Helmholtz':
            if self.helmholtz_locked():
                return
            try:
                radius = float(self.helmholtz_radius_var.get())
                if not np.isfinite(radius) or radius <= 0:
                    raise ValueError('Coil circle radius must be positive.')
                self.helmholtz_circle = (*point, radius)
                self.setup_revision += 1
                self.status.config(text='Align the blue circle with the Helmholtz coil. Its cross marks the force center. Save setup to retain it.')
                self.render()
            except ValueError as error:
                self.status.config(text=str(error))
            return
        if self.axis_pending is not None:
            self.axis_status.config(text="Use Capture settled robot, not a mouse click. Enlarge ROI radius if needed, or Reset axes to reposition the ROI.")
            return
        try:
            if self.mode.get() == "Corners":
                if len(self.corners) == 4:
                    self.status.config(text="Four points selected. Clear points to select again.")
                    return
                self.capture_pin(point, manual=self.manual_pin_var.get())
            else:
                radius = float(self.radius_var.get())
                if not np.isfinite(radius) or radius <= 0:
                    raise ValueError("ROI radius must be positive.")
                self.circle = (*point, radius)
                self.tracker = None
                self.auto_tracker_signature = None
                self.axis_tracker = None
                self.axis_detections = []
                self.axis_processed_at = None
                self.status.config(text='ROI placed. One-robot tracking starts automatically; pins are not required for detection.')
            self.detections = []
            self.render()
        except ValueError as error:
            self.status.config(text=f"Not saved: {error}")
            self.render()
            show_view_dialog(self, messagebox.showerror, "Calibration", str(error))

    def move_pin_roi(self, event, *, transform=None):
        popup = getattr(self, 'camera_popout', None)
        if transform is None and popup is not None and popup.is_open:
            return
        if getattr(self, 'command_running', False):
            return
        transform = self.transform if transform is None else transform
        if (self.frame is None or transform is None or self.mode.get() != "Corners"
                or self.axis_busy or self.axis_pending
                or (hasattr(self, "manual_pin_var") and self.manual_pin_var.get())):
            return
        ox, oy, scale = transform
        point = (round((event.x - ox) / scale), round((event.y - oy) / scale))
        h, w = self.frame.shape[:2]
        if not (0 <= point[0] < w and 0 <= point[1] < h):
            self.clear_pin_roi()
            return
        try:
            radius = float(self.pin_radius_var.get())
            if not np.isfinite(radius) or radius <= 0:
                return
            self.pin_search_circle = (*point, radius)
            self.render()
        except ValueError:
            pass

    def clear_pin_roi(self, event=None):
        if self.pin_search_circle is not None:
            self.pin_search_circle = None
            self.render()

    def on_pin_method_changed(self):
        self.clear_pin_roi()
        self.status.config(text=(
            "Manual pin placement: click the exact pin center. No green detection is performed."
            if self.manual_pin_var.get() else
            "Automatic pin detection: move the small ROI over one green pin, then click."
        ))

    def capture_pin(self, point, *, manual=False):
        """Use detected centers unless the user explicitly selects manual mode."""
        if len(self.corners) >= 4:
            raise ValueError("Four pins are already captured. Clear points to start again.")
        if manual:
            h, w = self.frame.shape[:2]
            center = np.asarray(point, dtype=float)
            if center.shape != (2,) or not np.isfinite(center).all() or np.any(center < 0) or np.any(center >= [w, h]):
                raise ValueError("Manual pin must be inside the camera frame.")
            self.store_pin(tuple(center), manual=True)
            return
        radius = float(self.pin_radius_var.get())
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("Pin search radius must be a positive finite number.")
        self.pin_search_circle = (*point, radius)
        mask = circle_detection_mask(self.frame.shape[:2], self.pin_search_circle)
        _, dots = self.detector.process(
            self.frame, mode="Color blobs", color="Green", minimum_area=1,
            morphology_kernel_size=1, draw_annotations=False, detection_mask=mask,
        )
        if not dots:
            raise ValueError("No green pin inside the small ROI. Move it over the pin or increase Pin search radius. Alternatively, enable Manual pin placement and click the exact pin center.")
        if len(dots) != 1:
            raise ValueError(f"Found {len(dots)} green candidates. Move or shrink the small ROI to include only one pin, then retry.")
        center = dots[0]["center"]
        self.store_pin(center, manual=False)

    def store_pin(self, center, *, manual):
        if any(np.linalg.norm(np.asarray(center) - old) < 3 for old in self.corners):
            raise ValueError("This pin is already captured. Move the small ROI to a different pin.")
        corners = [*self.corners, center]
        if len(corners) == 4:
            corners = OfflineDetectionGUI.order_corner_points(corners)
            calibrated_dwell_coordinates([], corners, float(self.width_var.get()), float(self.height_var.get()))
        self.corners = corners
        self.reset_axes()
        method = "manually placed" if manual else "automatically detected"
        self.status.config(text=f"Pin {method}. {len(corners)}/4 pins saved. " + (
            "Four-point calibration ready; place the robot ROI next." if len(corners) == 4
            else ("Click the exact center of the next pin." if manual else "Move the small ROI to the next green pin and click.")))

    def clear_points(self):
        self.reset_axes()
        self.corners = []
        self.pin_search_circle = None
        self.render()
        self.status.config(text="Select the four green corner dots.")

    def stop_tracking(self):
        if self.tracker is not None:
            self.circle = (*self.tracker.center, self.tracker.radius)
        self.tracker = None
        self.status.config(text="Tracking stopped. Coordinates remain available for export.")

    def render(self, canvas=None):
        self.refresh_helmholtz_center()
        if canvas is None:
            self.render_revision = getattr(self, 'render_revision', 0) + 1
        if hasattr(self, 'step_labels'):
            self.refresh_step_status(self.step_labels)
        if hasattr(self, 'capture_hint'):
            ready = bool(self.axis_pending) and not self.axis_busy and not self.command_running
            self.capture_hint.config(text=self.capture_explanation(),
                                     background='#e4f5e9' if ready else '#fff4d6',
                                     foreground='#176637' if ready else '#855300')
        if canvas is None and not getattr(self, 'visible', lambda: True)():
            return
        popup = getattr(self, 'camera_popout', None)
        if canvas is None and popup is not None and popup.is_open:
            self.canvas.delete('all')
            self.canvas.create_text(max(1, self.canvas.winfo_width()) / 2, max(1, self.canvas.winfo_height()) / 2,
                                    text='Camera is displayed in the pop-out window', fill='#c5d9ee')
            return
        main_view = canvas is None
        canvas = self.canvas if main_view else canvas
        if self.frame is None:
            return
        h, w = self.frame.shape[:2]
        cw, ch = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
        scale = min(cw / w, ch / h)
        dw, dh = max(1, round(w * scale)), max(1, round(h * scale))
        annotated = cv2.resize(self.frame, (dw, dh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

        def screen_point(point):
            return tuple(np.rint(np.asarray(point) * scale).astype(int))

        def stroke(draw, *args, color, width=1, **kwargs):
            # Overlay only the display-sized buffer, never the full camera image.
            args = list(args)
            if draw == cv2.polylines:
                args[0] = [np.rint(points * scale).astype(np.int32) for points in args[0]]
            elif draw == cv2.putText:
                args[1] = screen_point(args[1])
                kwargs['fontScale'] *= scale
            else:
                args[0] = screen_point(args[0])
                if draw == cv2.circle:
                    args[1] = max(1, round(args[1] * scale))
                elif draw == cv2.arrowedLine:
                    args[1] = screen_point(args[1])
                elif draw == cv2.drawMarker:
                    kwargs['markerSize'] = max(2, round(kwargs['markerSize'] * scale))
            kwargs["line_type" if draw in (cv2.drawMarker, cv2.arrowedLine) else "lineType"] = cv2.LINE_AA
            draw(annotated, *args, color=color,
                 thickness=max(1, round(width)), **kwargs)

        def marker(point, color):
            stroke(cv2.drawMarker, point, color=color, width=1,
                   markerType=cv2.MARKER_CROSS, markerSize=max(4, round(6 / scale)))

        def label(text, point, color):
            stroke(cv2.putText, text, point, color=color, width=1,
                   fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.32 / scale)

        mask_var = getattr(self, "show_mask_var", None)
        if mask_var is not None and mask_var.get() and (self.circle is not None or self.tracker is not None):
            circle = self.circle if self.tracker is None else (*self.tracker.center, self.tracker.radius)
            mask = circle_detection_mask(self.frame.shape[:2], circle)
            if len(self.corners) == 4:
                workspace = workspace_detection_mask(self.frame.shape[:2], self.corners)
                mask = workspace if mask is None else cv2.bitwise_and(mask, workspace)
            try:
                key = (self.frame_received_at, tuple(circle) if circle is not None else None,
                       tuple(map(tuple, self.corners)), float(self.area_var.get()), self.read_contrast_threshold())
                if getattr(self, 'mask_preview_key', None) != key:
                    _, candidates = self.detector.process(
                        self.frame, mode="Color blobs", color="Dark contrast",
                        minimum_area=key[3], morphology_kernel_size=1,
                        draw_annotations=False, detection_mask=mask, dark_contrast_threshold=key[4],
                    )
                    candidates, _ = exclude_calibration_markers(candidates, self.corners, 10)
                    self.mask_preview = (self.detector.last_blob_mask.copy(), candidates)
                    self.mask_preview_key = key
                binary_mask, candidates = self.mask_preview
                annotated = cv2.cvtColor(cv2.resize(binary_mask, (dw, dh), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
                for item in candidates:
                    marker(item["center"], (0, 180, 255))
                self.detection_info.config(text=f"{len(candidates)} eligible blobs; white = contrast pixels")
            except (ValueError, TypeError):
                self.detection_info.config(text="Enter valid contrast and minimum area values.")
        elif hasattr(self, "detection_info"):
            self.detection_info.config(text="")
        pin_circle = getattr(self, "pin_search_circle", None)
        coil_circle = getattr(self, 'helmholtz_circle', None)
        if coil_circle is not None:
            center = tuple(round(v) for v in coil_circle[:2])
            stroke(cv2.circle, center, round(coil_circle[2]), color=(255, 180, 40), width=2)
            marker(center, (255, 180, 40))
            label('Helmholtz axis', (center[0]+round(10/scale), center[1]-round(10/scale)), (255, 180, 40))
        if pin_circle is not None and self.mode.get() == "Corners":
            stroke(cv2.circle, tuple(round(v) for v in pin_circle[:2]), round(pin_circle[2]), color=(255, 255, 0))
        if len(self.corners) == 4:
            stroke(cv2.polylines, [np.asarray(self.corners, np.int32)], True, color=(255, 0, 255))
        for index, point in enumerate(self.corners, 1):
            p = tuple(map(int, point))
            marker(p, (255, 0, 255))
            offset = max(5, round(10 / scale))
            label(f"P{index}", (p[0] + offset, max(offset, p[1] - offset)), (255, 0, 255))
        circle = self.circle if self.tracker is None else (*self.tracker.center, self.tracker.radius)
        if circle is not None:
            stroke(cv2.circle, tuple(map(lambda v: round(float(v)), circle[:2])),
                   round(float(circle[2])), color=(0, 255, 255))
        for item in self.detections:
            marker(item["center"], (0, 0, 255))
        samples = getattr(self, "axis_samples", {})
        if "Origin" in samples:
            origin = tuple(map(int, samples["Origin"]))
            marker(origin, (255, 255, 255))
            label("O", (origin[0] + round(10 / scale), origin[1] + round(18 / scale)), (255, 255, 255))
            axis_points = samples
            if len(samples) == 3:
                axes = RobotAxes(self.corners, float(self.width_var.get()), float(self.height_var.get()),
                                 samples, getattr(self, 'axis_capture_moves', {}))
                axis_points = dict(zip(AXIS_TARGETS_M, axes.pixel_positions([[0, 0], [1, 0], [0, 1]])))
            for name, color in (("+X", (0, 0, 255)), ("+Y", (255, 180, 0))):
                if name in samples:
                    point = tuple(map(int, axis_points[name]))
                    delta = np.asarray(point) - origin
                    length = np.linalg.norm(delta)
                    end = tuple(np.rint(np.asarray(origin) + delta / max(length, 1) * 45 / scale).astype(int))
                    stroke(cv2.arrowedLine, origin, end, color=color, tipLength=.15)
                    stroke(cv2.circle, point, max(3, round(3 / scale)), color=color, width=1)
                    label(name, end, color)
        if self.manual_lines_method():
            points = getattr(self, 'axis_line_points', {})
            for negative, positive, color in (('-X', '+X', (0, 0, 255)), ('-Y', '+Y', (255, 180, 0))):
                if negative in points and positive in points:
                    stroke(cv2.line, tuple(np.rint(points[negative]).astype(int)),
                           tuple(np.rint(points[positive]).astype(int)), color=color)
                for name in (negative, positive):
                    if name in points:
                        point = tuple(np.rint(points[name]).astype(int))
                        marker(point, color)
                        label(name+' line', tuple(np.rint(np.asarray(point)+8/scale).astype(int)), color)
        if self.symmetric_method():
            for name, point in getattr(self, 'axis_raw_captures', {}).items():
                marker(point, (180, 255, 180))
                label(AXIS_DISPLAY_NAMES[name], tuple(np.rint(np.asarray(point) + 8 / scale).astype(int)), (180, 255, 180))
        canvas.camera_transform = ((cw - dw) / 2, (ch - dh) / 2, scale)
        if main_view:
            self.transform = canvas.camera_transform
        image = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        canvas.camera_photo = ImageTk.PhotoImage(image)
        if main_view:
            self.photo = canvas.camera_photo
        canvas.delete("all")
        canvas.create_image(cw / 2, ch / 2, image=canvas.camera_photo, anchor="center")

    def setup_data(self, *, require_capture=True, require_roi=True):
        if not require_capture and getattr(self, "axis_busy", False):
            raise ValueError("Wait for the calibration motor move to finish before using the setup in Commands.")
        if require_capture and (getattr(self, "axis_busy", False) or getattr(self, "axis_pending", None)):
            raise ValueError("Capture the current settled robot position before saving.")
        manual = self.manual_lines_method()
        if manual and len(getattr(self, 'axis_line_points', {})) != 4:
            raise ValueError('Complete manual axis lines: click -X, +X, -Y, +Y before saving or using the setup.')
        if self.frame is None or len(self.corners) != 4:
            raise ValueError('Select a frame and four workspace points first.')
        if self.circle is None and require_roi and not (manual and require_capture):
            raise ValueError('Place a robot ROI in Calibration before using the setup for tracking or Commands.')
        data = {
            "version": 2, "video_size_px": list(self.frame.shape[1::-1]),
            "helmholtz_circle_px": getattr(self, 'helmholtz_circle', None),
            "helmholtz_center_xy_m": self.helmholtz_center_m(),
            "corners_px": self.corners, "rectangle_width_cm": float(self.width_var.get()),
            "rectangle_height_cm": float(self.height_var.get()),
            "roi_circle_px": None if self.circle is None else [
                *(self.circle[:2] if self.tracker is None else self.tracker.center),
                float(self.radius_var.get()) if self.tracker is None else self.tracker.radius,
            ],
            "robot_count": 1, "minimum_area": float(self.area_var.get()),
            'camera_source': getattr(self, 'camera_source_key', 'Pi SSH'),
            "dark_contrast_threshold": self.read_contrast_threshold(),
            "axis_method": "manual_lines" if manual else "symmetric" if self.symmetric_method() else "center",
            "axis_line_points_px": getattr(self, "axis_line_points", {}) if manual else {},
            "axis_raw_captures_px": getattr(self, "axis_raw_captures", {}),
            "axis_samples_px": getattr(self, "axis_samples", {}),
            "axis_capture_moves": {key: getattr(self, 'axis_capture_moves', {}).get(key, key)
                                   for key in getattr(self, 'axis_samples', {})},
            "axis_targets_m": {key: AXIS_SAVED_TARGETS_M[getattr(self, 'axis_capture_moves', {}).get(key, key)]
                               for key in AXIS_TARGETS_M},
            "control_case": hardware_case_name(self.control_case_var.get()) if hasattr(self, "control_case_var") else "",
            "coordinate_system": "robot_axes" if len(getattr(self, "axis_samples", {})) == 3 else "rectangle_center",
        }
        if self.symmetric_method() and self.axis_raw_captures and not self.axis_samples:
            raise ValueError(self.axis_instructions())
        if data["axis_samples_px"] and len(data["axis_samples_px"]) != 3:
            raise ValueError("Capture Center, +X and +Y or Reset axes before saving. Incomplete axes cannot be saved.")
        if data["roi_circle_px"] is None:
            RobotAxes(data["corners_px"], data["rectangle_width_cm"], data["rectangle_height_cm"], data["axis_samples_px"])
        else:
            MovingROICalibration(data["corners_px"], data["rectangle_width_cm"],
                                 data["rectangle_height_cm"], data["roi_circle_px"],
                                 data["robot_count"], data["minimum_area"], data["axis_samples_px"],
                                 contrast_threshold=data["dark_contrast_threshold"], axis_capture_moves=data["axis_capture_moves"])
        self.refresh_helmholtz_center()
        return data

    def save_setup(self):
        try:
            data = self.setup_data()
            SETUP_DIR.mkdir(parents=True, exist_ok=True)
            path = show_view_dialog(self, filedialog.asksaveasfilename,
                title="Save GUI005 calibration", initialdir=str(SETUP_DIR),
                initialfile=default_setup_name(), defaultextension=".json",
                filetypes=[("Calibration setup", "*.json")],
            )
            if path:
                with open(path, "w", encoding="utf-8") as stream:
                    json.dump(data, stream, indent=2)
                self.last_saved_setup_path = Path(path)
                self.status.config(text='Manual axis calibration saved. A robot ROI can be placed later.' if data['roi_circle_px'] is None else
                                   'Complete setup saved: four points, physical dimensions, ROI, detection settings, hardware case and XY calibration.')
        except (ValueError, OSError) as error:
            show_view_dialog(self, messagebox.showerror, "Calibration", str(error))

    def load_last_setup(self):
        if self.frame is None:
            show_view_dialog(self, messagebox.showerror, "Calibration", "Wait for a live camera frame first.")
            return
        try:
            candidates = list(SETUP_DIR.glob('*.json'))
            saved = getattr(self, 'last_saved_setup_path', None)
            if saved is not None and Path(saved).is_file():
                candidates.append(Path(saved))
            if not candidates:
                raise ValueError('No saved setup found. Save a setup or use Load setup first.')
            latest = max(candidates, key=lambda path: path.stat().st_mtime_ns)
        except (ValueError, OSError) as error:
            show_view_dialog(self, messagebox.showerror, "Calibration", str(error))
            return
        self.load_setup(latest)

    def load_setup(self, path=None):
        if self.frame is None:
            show_view_dialog(self, messagebox.showerror, "Calibration", "Wait for a live camera frame first.")
            return
        if path is None:
            SETUP_DIR.mkdir(parents=True, exist_ok=True)
            path = show_view_dialog(self, filedialog.askopenfilename,
                title="Load GUI005 calibration", initialdir=str(SETUP_DIR),
                filetypes=[("JSON", "*.json")],
            )
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as stream:
                data = json.load(stream)
            if data["version"] not in (1, 2) or data["video_size_px"] != list(self.frame.shape[1::-1]):
                raise ValueError("Calibration does not match the current frame size.")
            if data.get('camera_source') is not None and data['camera_source'] != getattr(self, 'camera_source_key', 'Pi SSH'):
                raise ValueError('This calibration belongs to a different camera source/device. Load a matching setup or recalibrate.')
            if data.get('camera_source') is None and getattr(self, 'camera_source_key', 'Pi SSH') != 'Pi SSH':
                raise ValueError('This older setup has no camera identity. Recalibrate and save a setup for the selected webcam.')
            samples = data.get("axis_samples_px", {})
            if samples and set(samples) != set(AXIS_TARGETS_M):
                raise ValueError("Saved robot axis references are incomplete.")
            raw_captures = data.get('axis_raw_captures_px', {})
            method = data.get('axis_method', 'center')
            if method not in ('center', 'symmetric', 'manual_lines'):
                raise ValueError('Saved axis calibration method is unsupported.')
            symmetric, manual = method == 'symmetric', method == 'manual_lines'
            line_points = data.get('axis_line_points_px', {})
            report = ''
            if symmetric and raw_captures and not samples:
                raise ValueError('Saved symmetric calibration is incomplete.')
            if symmetric and samples:
                if any(name not in AXIS_MOVE_TARGETS_M for name in raw_captures):
                    raise ValueError('Saved symmetric capture names are invalid.')
                if not isinstance(raw_captures, dict) or not all(n in raw_captures for n in ('-X', '+X', '-Y', '+Y')):
                    raise ValueError('Saved symmetric captures are incomplete.')
                samples, report = symmetric_axes(data['corners_px'], data['rectangle_width_cm'],
                                                 data['rectangle_height_cm'], raw_captures)
            if manual:
                if not isinstance(line_points, dict) or set(line_points) != {'-X', '+X', '-Y', '+Y'}:
                    raise ValueError('Saved manual axis lines are incomplete.')
                samples, report = manual_line_axes(data['corners_px'], data['rectangle_width_cm'],
                                                   data['rectangle_height_cm'], line_points)
            capture_moves = ({key: key for key in samples} if manual else
                             data.get('axis_capture_moves', {key: key for key in samples}))
            if (not isinstance(capture_moves, dict) or (samples and set(capture_moves) != set(samples))
                    or any(not isinstance(move, str) or AXIS_REFERENCE_NAMES.get(move) != key
                           for key, move in capture_moves.items())):
                raise ValueError('Saved axis move references are invalid.')
            roi = data['roi_circle_px']
            if roi is None and not manual:
                raise ValueError('Saved robot calibration needs an ROI.')
            # Validate geometry/settings through the same tracker constructor. This
            # temporary ROI is never restored or saved as an actual selected ROI.
            validation_roi = roi if roi is not None else [self.frame.shape[1]/2, self.frame.shape[0]/2, DEFAULT_ROBOT_ROI_RADIUS]
            tracker = MovingROICalibration(
                data["corners_px"], data["rectangle_width_cm"], data["rectangle_height_cm"],
                validation_roi, 1, data["minimum_area"],
                samples, axis_capture_moves=capture_moves,
                contrast_threshold=data.get("dark_contrast_threshold", 10),
            )
            h, w = self.frame.shape[:2]
            coil_circle = data.get('helmholtz_circle_px')
            if coil_circle is not None:
                coil_circle = np.asarray(coil_circle, float)
                if (coil_circle.shape != (3,) or not np.isfinite(coil_circle).all()
                        or coil_circle[2] <= 0 or np.any(coil_circle[:2] < 0)
                        or np.any(coil_circle[:2] >= [w, h])):
                    raise ValueError('Saved Helmholtz circle is invalid or outside the frame.')
                coil_circle = tuple(coil_circle)
            points = np.asarray([*tracker.corners, tracker.center,
                                 *(line_points.values() if manual else raw_captures.values() if symmetric else samples.values()),
                                 *([samples['Origin']] if manual else [])], dtype=float)
            if not np.isfinite(points).all():
                raise ValueError('Saved calibration points must be finite.')
            if np.any(points < 0) or np.any(points >= [w, h]):
                raise ValueError("Calibration points or ROI center are outside the frame.")
        except (ValueError, TypeError, KeyError, OSError) as error:
            show_view_dialog(self, messagebox.showerror, "Calibration", str(error))
            return
        self.reset_axes()
        self.clear_tracking()
        self.corners, self.circle = tracker.corners, None if roi is None else tuple(roi)
        self.helmholtz_circle = coil_circle
        if coil_circle is not None:
            self.helmholtz_radius_var.set(str(coil_circle[2]))
        self.pin_search_circle = None
        for variable, value in ((self.width_var, tracker.width_cm), (self.height_var, tracker.height_cm),
                                (self.radius_var, tracker.radius), (self.count_var, tracker.count),
                                (self.area_var, tracker.minimum_area)):
            variable.set(str(value))
        self.control_case_var.set(hardware_case_name(data.get("control_case", "")))
        if hasattr(self, "contrast_var"):
            self.contrast_var.set(str(tracker.contrast_threshold))
        if hasattr(self, 'axis_method_var'):
            self.axis_method_var.set('Manual axis lines' if manual else 'Symmetric pairs' if symmetric else 'Center reference')
            self.axis_method_name = self.axis_method_var.get()
        self.axis_raw_captures = raw_captures if symmetric else {}
        self.axis_line_points = line_points if manual else {}
        if manual:
            self.mode.set('AxisLines')
        elif getattr(self, 'mode', None) is not None and self.mode.get() == 'AxisLines':
            self.mode.set('ROI')
        self.refresh_axis_method_controls()
        self.axis_samples = samples
        self.axis_capture_moves = capture_moves if samples else {}
        self.axis_status.config(text=("Manual axis lines restored. " + report if manual else "Symmetric calibration restored. " + report if symmetric else "Saved robot origin and XY axes restored.") if samples else "Setup has image-aligned XY. Calibrate robot axes if needed.")
        self.setup_revision = getattr(self, 'setup_revision', 0) + 1
        self.render()
        self.status.config(text='Setup loaded. Place a robot ROI when ready to track.' if roi is None else
                                'Setup loaded. One-robot tracking resumes automatically from the ROI.')

    def export_csv(self):
        if not self.rows:
            show_view_dialog(self, messagebox.showerror, "Coordinates", "Place an ROI and wait for live frames to collect measurements first.")
            return
        path = show_view_dialog(self, filedialog.asksaveasfilename, title="Export robot XY", defaultextension=".csv")
        if path:
            try:
                with open(path, "w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["frame", "time_s", "group_valid", "robot_id", "x_px", "y_px",
                                     "x_cm", "y_cm", "group_x_cm", "group_y_cm", "roi_x_px", "roi_y_px"])
                    writer.writerows(self.rows)
                self.status.config(text="Robot and group XY coordinates exported.")
            except OSError as error:
                show_view_dialog(self, messagebox.showerror, "Coordinates", str(error))

    def close(self):
        if getattr(self, 'prepare_after', None) is not None:
            self.parent.after_cancel(self.prepare_after)
            self.prepare_after = None
        self.prepare_job = getattr(self, 'prepare_job', 0) + 1
        if getattr(self, 'camera_popout', None) is not None:
            self.camera_popout.close()
        self.axis_job += 1
        self.axis_busy = False
        if self.axis_after is not None:
            self.parent.after_cancel(self.axis_after)
            self.axis_after = None
        self.axis_tracker = None
        self.axis_detections = []
        self.axis_processed_at = None
        self.clear_tracking()
