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
from axis_calibration import AXIS_TARGETS_M, AXIS_DISPLAY_NAMES, RobotAxes
from point_control import hardware_case_name
from camera_popout import CameraPopout


SETUP_DIR = Path(__file__).resolve().parent / "setups"
DEFAULT_ROBOT_ROI_RADIUS = 300
DEFAULT_PIN_ROI_RADIUS = 35


def default_setup_name():
    return datetime.now(ZoneInfo("Asia/Tehran")).strftime("%Y%m%d_%H%M%S.json")


class MovingROICalibration:
    """Frame-by-frame tracking shared by live preview and movie playback."""

    def __init__(self, corners, width_cm, height_cm, circle, count, minimum_area, axis_samples=None,
                 contrast_threshold=10, strict_single=False, limit_to_pins=True):
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
        self.limit_to_pins = limit_to_pins
        self.selection_error = None
        self.detector = RobotDetector()
        self.axes = RobotAxes(corners, width_cm, height_cm, axis_samples) if axis_samples else None
        self.contrast_threshold = float(contrast_threshold)
        if not np.isfinite(self.contrast_threshold) or not 1 <= self.contrast_threshold <= 255:
            raise ValueError("Dark contrast threshold must be between 1 and 255.")

    def coordinates(self, points):
        if self.axes is not None:
            return self.axes.coordinates(points)
        return calibrated_dwell_coordinates(
            [{"center": point} for point in points], self.corners, self.width_cm, self.height_cm,
        )

    def process(self, frame):
        mask = circle_detection_mask(frame.shape[:2], (*self.center, self.radius))
        if self.limit_to_pins:
            mask = cv2.bitwise_and(mask, workspace_detection_mask(frame.shape[:2], self.corners))
        _, candidates = self.detector.process(
            frame, mode="Color blobs", color="Dark contrast",
            minimum_area=self.minimum_area, morphology_kernel_size=1,
            draw_annotations=False, detection_mask=mask,
            dark_contrast_threshold=self.contrast_threshold,
        )
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
        xy = self.coordinates([item["center"] for item in selected])
        for item, coordinates in zip(selected, xy):
            item["xy_cm"] = tuple(coordinates)
        group_xy = None if center is None else tuple(self.coordinates([center])[0])
        return selected, center, group_xy


class CalibrationTab:
    def __init__(self, parent, axis_move=None):
        self.parent = parent
        self.frame = None
        self.corners = []
        self.circle = None
        self.pin_search_circle = None
        self.transform = None
        self.tracker = None
        self.detections = []
        self.capture = None
        self.frame_number = 0
        self.fps = 30.0
        self.playing = False
        self.frozen = False
        self.rows = []
        self.live_started = None
        self.after_id = None
        self.source = "Live Pi camera"
        self.detector = RobotDetector()
        self.axis_move = axis_move
        self.axis_samples = {}
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

        bar = tk.Frame(parent)
        bar.pack(fill="x", padx=8, pady=6)
        for label, command in (
            ("Live camera", self.use_live), ("Open movie", self.open_movie),
            ("Play / Pause", self.toggle_play), ("First frame", self.rewind),
            ("Save setup", self.save_setup), ("Load setup", self.load_setup),
            ("Export XY CSV", self.export_csv),
            ("Pop-out camera", self.open_camera_popout),
        ):
            tk.Button(bar, text=label, command=command).pack(side="left", padx=2)
        settings = tk.Frame(parent)
        settings.pack(fill="x", padx=8, pady=3)
        self.width_var = tk.StringVar(value="10")
        self.height_var = tk.StringVar(value="10")
        self.radius_var = tk.StringVar(value=str(DEFAULT_ROBOT_ROI_RADIUS))
        self.pin_radius_var = tk.StringVar(value=str(DEFAULT_PIN_ROI_RADIUS))
        self.manual_pin_var = tk.BooleanVar(value=False)
        self.count_var = tk.StringVar(value="3")
        self.area_var = tk.StringVar(value="1")
        self.contrast_var = tk.StringVar(value="10")
        self.show_mask_var = tk.BooleanVar(value=False)
        for label, variable in (("Width (cm)", self.width_var),
                                ("Height (cm)", self.height_var),
                                ("ROI radius (px)", self.radius_var),
                                ("Robots", self.count_var),
                                ("Min area (px²)", self.area_var)):
            tk.Label(settings, text=label).pack(side="left", padx=(4, 2))
            tk.Entry(settings, textvariable=variable, width=6).pack(side="left")
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
        tk.Button(tools, text="Freeze / Resume live", command=self.freeze_live).pack(side="left", padx=4)
        tk.Button(tools, text="Start tracking", command=self.start_tracking).pack(side="left", padx=4)
        tk.Button(tools, text="Stop tracking", command=self.stop_tracking).pack(side="left", padx=4)
        pin_tools = tk.Frame(parent)
        pin_tools.pack(fill="x", padx=8, pady=2)
        tk.Label(pin_tools, text="Pin search radius (px)").pack(side="left")
        tk.Spinbox(pin_tools, from_=3, to=300, textvariable=self.pin_radius_var, width=5).pack(side="left", padx=4)
        tk.Checkbutton(pin_tools, text="Manual pin placement", variable=self.manual_pin_var,
                       command=self.on_pin_method_changed).pack(side="left", padx=4)
        tk.Label(pin_tools, text="Auto: click the small ROI over a green pin. Manual: click the exact pin center.").pack(side="left", padx=4)
        axes_bar = tk.LabelFrame(parent, text="Robot XY axes — use one robot", padx=5, pady=4)
        axes_bar.pack(fill="x", padx=8, pady=4)
        axes_bar.columnconfigure(1, weight=1)
        self.control_case_var = tk.StringVar(value="")
        cases_dir = Path(__file__).resolve().parents[1] / "cases"
        cases = [path.parent.name for path in sorted(cases_dir.glob("case_*/case.py"))]
        tk.Label(axes_bar, text="Hardware case").grid(row=0, column=0)
        ttk.Combobox(axes_bar, textvariable=self.control_case_var, values=cases,
                     width=39, state="readonly").grid(row=0, column=1, padx=4, sticky="ew")
        self.axis_wait_var = tk.StringVar(value="3")
        tk.Label(axes_bar, text="Settle (s)").grid(row=0, column=2)
        tk.Entry(axes_bar, textvariable=self.axis_wait_var, width=4).grid(row=0, column=3)
        actions = tk.Frame(axes_bar)
        actions.grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 0))
        self.axis_buttons = []
        for name in AXIS_TARGETS_M:
            button = tk.Button(actions, text=f"Move to {AXIS_DISPLAY_NAMES[name]}", command=lambda n=name: self.move_axis(n))
            button.pack(side="left", padx=3)
            self.axis_buttons.append(button)
        self.axis_capture_button = tk.Button(actions, text="Capture settled robot", command=self.capture_axis, state="disabled")
        self.axis_capture_button.pack(side="left", padx=3)
        tk.Button(actions, text="Reset axes", command=self.reset_axes).pack(side="left", padx=3)
        tk.Label(axes_bar, anchor="w", justify="left", text=(
            "Choose the case matching your actual magnet count, ring radius, magnet sizes and magnetic properties. "
            "Only the case.py hardware model is used; no condition or sequence file is needed."
        ), wraplength=1000).grid(row=2, column=0, columnspan=4, sticky="ew", pady=(4, 0))
        self.axis_status = tk.Label(axes_bar, anchor="w", text=(
            "Ratio = 1. Place the ROI around one robot; it follows the robot during moves. Move to Center, +X (10 mm), then +Y (10 mm). "
            "After each move, visually confirm the robot has settled and press Capture settled robot. No mouse marking."
        ), wraplength=1000)
        self.axis_status.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(4, 0))
        tk.Label(parent, anchor="w", text=(
            "Select four corner dots, enter their rectangle size, then place a circular ROI around all robots. "
            "Dark contrast detection only. Calibrate robot axes below to replace the default image-aligned XY."
        ), wraplength=1000).pack(fill="x", padx=10)
        self.status = tk.Label(parent, text="Waiting for a live camera frame or movie.", anchor="w")
        self.status.pack(fill="x", padx=10, pady=4)
        self.canvas = tk.Canvas(parent, background="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.canvas.bind("<Button-1>", self.click)
        self.canvas.bind("<Motion>", self.move_pin_roi)
        self.canvas.bind("<Leave>", self.clear_pin_roi)
        self.canvas.bind("<Configure>", lambda event: self.render())
        for variable in (self.width_var, self.height_var, self.control_case_var):
            variable.trace_add("write", self.on_metric_size_changed)

    def on_metric_size_changed(self, *args):
        if self.axis_samples or self.axis_busy or self.axis_pending or getattr(self, "axis_tracker", None) is not None:
            self.reset_axes()
            self.axis_status.config(text="Rectangle size or control case changed. Recalibrate the robot axes.")

    def clear_tracking(self):
        self.tracker = None
        self.detections = []
        self.rows = []
        self.live_started = None

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
        self.axis_tracker = None
        self.axis_detections = []
        self.axis_processed_at = None
        for button in self.axis_buttons:
            button.config(state="normal")
        if hasattr(self, "axis_capture_button"):
            self.axis_capture_button.config(state="disabled")
        self.axis_status.config(text="Axes reset. Move to Center, then capture the settled robot in the ROI. Repeat for +X and +Y. Ratio is 1.")
        self.render()

    def move_axis(self, name):
        if self.axis_busy:
            return
        if self.axis_pending:
            messagebox.showerror("Robot axes", "Capture the current settled position before another move, or Reset axes to cancel.")
            return
        try:
            if self.capture is not None:
                raise ValueError("Axis movement requires the live Pi camera.")
            if self.frame is None or self.circle is None or len(self.corners) != 4:
                raise ValueError("Select four corners and place an ROI around the single robot first.")
            if int(self.count_var.get()) != 1:
                raise ValueError("Set Robots to 1 for axis calibration.")
            if name != "Origin" and "Origin" not in self.axis_samples:
                raise ValueError("Move to Center and capture it first.")
            if not self.control_case_var.get():
                raise ValueError("Select the control case matching your hardware geometry first.")
            wait = float(self.axis_wait_var.get())
            if not np.isfinite(wait) or wait <= 0:
                raise ValueError("Settle time must be positive.")
            if self.axis_move is None:
                raise ValueError("Controller movement is unavailable.")
            calibrated_dwell_coordinates([], self.corners, float(self.width_var.get()), float(self.height_var.get()))
            if self.axis_tracker is None:
                circle = self.circle if self.tracker is None else (*self.tracker.center, self.tracker.radius)
                self.axis_tracker = MovingROICalibration(
                    self.corners, float(self.width_var.get()), float(self.height_var.get()),
                    (*circle[:2], float(self.radius_var.get())), 1, float(self.area_var.get()),
                    contrast_threshold=self.read_contrast_threshold(), strict_single=True,
                )
            self.update_axis_tracking()
            if not self.axis_detections:
                raise ValueError(self.axis_tracker.selection_error or "Robot not acquired. Check Show detection mask and the ROI before commanding a move.")
        except ValueError as error:
            messagebox.showerror("Robot axes", str(error))
            return
        self.stop_tracking()
        self.clear_tracking()
        if name == "Origin":
            self.axis_samples = {}
        self.axis_pending = None
        self.axis_busy = True
        self.axis_capture_button.config(state="disabled")
        self.axis_job += 1
        job = self.axis_job
        case = self.control_case_var.get()
        self.frozen = False
        for button in self.axis_buttons:
            button.config(state="disabled")
        target = AXIS_TARGETS_M[name]
        self.axis_status.config(text=f"Calculating {AXIS_DISPLAY_NAMES[name]} {target} m with ratio 1, then moving and settling for {wait:g} s...")

        def worker():
            try:
                self.axis_move(target, case, wait, lambda: self.axis_job == job)
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
                    messagebox.showerror("Robot axes", error)
                else:
                    self.axis_pending = name
                    self.axis_ready_at = time.monotonic()
                    for button in self.axis_buttons:
                        button.config(state="disabled")
                    self.axis_capture_button.config(state="normal")
                    self.axis_status.config(text=f"{AXIS_DISPLAY_NAMES[name]} commanded. Confirm the robot has stopped inside the ROI, then press Capture settled robot. The wait alone does not prove it settled.")
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
            messagebox.showerror("Robot axes — capture not saved", str(error))

    def record_axis_from_roi(self):
        if self.axis_busy or not self.axis_pending:
            raise ValueError("Move to a calibration position first and wait for the command to finish.")
        if self.capture is not None or self.frozen:
            raise ValueError("Capture needs the running live camera, not a movie or frozen frame.")
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
        if self.axis_pending != "Origin" and "Origin" not in self.axis_samples:
            raise ValueError("Capture Center first.")
        if self.axis_pending != "Origin" and np.linalg.norm(np.asarray(center) - self.axis_samples["Origin"]) < 2:
            raise ValueError("Robot has not moved enough from Center (less than 2 pixels). Check the movement and wait for it to settle; nothing was saved.")
        samples = dict(self.axis_samples, **{self.axis_pending: list(center)})
        if len(samples) == 3:
            axes = RobotAxes(self.corners, float(self.width_var.get()), float(self.height_var.get()), samples)
            self.axis_status.config(text=(
                f"Robot axes calibrated. Observed X move: {axes.lengths_cm[0] * 10:.2f} mm; "
                f"Y move: {axes.lengths_cm[1] * 10:.2f} mm. Save setup to retain this calibration."
            ))
        else:
            self.axis_status.config(text=f"{AXIS_DISPLAY_NAMES[self.axis_pending]} captured at ({center[0]:.1f}, {center[1]:.1f}) px. {len(samples)}/3 references recorded. Move to the next position.")
        self.axis_samples = samples
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

    def close_capture(self):
        self.playing = False
        self.axis_tracker = None
        self.axis_detections = []
        self.axis_processed_at = None
        if self.after_id is not None:
            self.parent.after_cancel(self.after_id)
            self.after_id = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def use_live(self):
        if self.axis_busy or self.axis_pending:
            self.reset_axes()
        self.close_capture()
        self.source = "Live Pi camera"
        self.frozen = False
        self.clear_tracking()
        self.status.config(text="Live camera selected. Connect the Pi in Connections.")

    def open_movie(self):
        path = filedialog.askopenfilename(title="Calibration movie", filetypes=[
            ("Movies", "*.mp4 *.avi *.mov *.mkv *.m4v"), ("All files", "*.*")])
        if not path:
            return
        capture = cv2.VideoCapture(path)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            messagebox.showerror("Movie", "Could not read the first movie frame.")
            return
        self.close_capture()
        self.capture, self.source = capture, path
        if self.axis_busy or self.axis_pending:
            self.reset_axes()
        fps = capture.get(cv2.CAP_PROP_FPS)
        self.fps = fps if np.isfinite(fps) and fps > 0 else 30.0
        self.frame_number = 0
        self.clear_tracking()
        self.accept_frame(frame)
        self.status.config(text="Movie frame 0. Select four dots and place the ROI, then Start tracking.")

    def accept_live_frame(self, frame):
        if self.capture is None and not self.frozen:
            self.accept_frame(frame)

    def accept_frame(self, frame):
        if self.frame is not None and self.frame.shape[:2] != frame.shape[:2]:
            self.corners, self.circle = [], None
            self.pin_search_circle = None
            self.reset_axes()
            self.clear_tracking()
            self.status.config(text="Frame size changed. Select four points and place the ROI again.")
        self.frame = frame.copy()
        self.frame_received_at = time.monotonic()
        if getattr(self, "axis_tracker", None) is not None and self.tracker is None:
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
                elapsed = (self.frame_number / self.fps if self.capture is not None
                           else time.monotonic() - self.live_started)
                valid = center is not None
                for item in self.detections or [None]:
                    self.rows.append([
                        self.frame_number, elapsed, int(valid),
                        "" if item is None else item["robot_id"],
                        "" if item is None else item["center"][0],
                        "" if item is None else item["center"][1],
                        "" if item is None else item["xy_cm"][0],
                        "" if item is None else item["xy_cm"][1],
                        "" if xy is None else xy[0], "" if xy is None else xy[1],
                        *self.tracker.center,
                    ])
                detail = "Group missing: ROI held at last valid center."
                if valid:
                    detail = f"Group XY = ({xy[0]:.3f}, {xy[1]:.3f}) cm"
                self.status.config(text=f"Frame {self.frame_number}: {len(self.detections)}/{self.tracker.count} robots. {detail}")
            except Exception as error:
                self.stop_tracking()
                self.status.config(text=f"Tracking stopped: {error}")
        self.render()
        if self.capture is None:
            self.frame_number += 1

    def toggle_play(self):
        if self.capture is None:
            return
        self.playing = not self.playing
        if self.playing and self.after_id is None:
            self.movie_tick()

    def movie_tick(self):
        self.after_id = None
        if not self.playing or self.capture is None:
            return
        started = time.monotonic()
        ok, frame = self.capture.read()
        if not ok:
            self.playing = False
            self.status.config(text="Movie complete. Export XY CSV to save the tracked coordinates.")
            return
        self.frame_number += 1
        self.accept_frame(frame)
        delay = max(1, round(1000 * (1 / self.fps - (time.monotonic() - started))))
        self.after_id = self.parent.after(delay, self.movie_tick)

    def rewind(self):
        if self.capture is None:
            return
        self.playing = False
        self.clear_tracking()
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = self.capture.read()
        if ok:
            self.frame_number = 0
            self.accept_frame(frame)
            self.status.config(text="First frame. Place ROI on the initial robots, then Start tracking.")

    def freeze_live(self):
        if self.axis_busy or self.axis_pending:
            self.axis_status.config(text="Keep the live preview running until the robot measurement is recorded.")
            return
        if self.capture is None:
            self.frozen = not self.frozen
            self.status.config(text="Live frame frozen for setup." if self.frozen else "Live preview resumed.")

    def open_camera_popout(self):
        if self.camera_popout is not None and self.camera_popout.is_open:
            self.camera_popout.show()
            return
        self.camera_popout = CameraPopout(
            self.parent, 'GUI005 — Calibration camera', self.render,
            actions=(('Center', lambda: self.move_axis('Origin'), self.axis_buttons[0]),
                     ('+X', lambda: self.move_axis('+X'), self.axis_buttons[1]),
                     ('+Y', lambda: self.move_axis('+Y'), self.axis_buttons[2]),
                     ('Capture', self.capture_axis, self.axis_capture_button),
                     ('Reset axes', self.reset_axes, None),
                     ('Clear pins', self.clear_points, None),
                     ('Track', self.start_tracking, None),
                     ('Stop track', self.stop_tracking, None),
                     ('Freeze', self.freeze_live, None)),
            toolbar_builder=self.build_popout_tools, mirror_action_text=False,
            controls_enabled=lambda: not self.command_running,
            status=lambda: f"{self.status.cget('text')}  {self.axis_status.cget('text')}",
        )
        canvas = self.camera_popout.canvas
        canvas.bind('<Button-1>', lambda event: self.click(event, transform=getattr(canvas, 'camera_transform', None)))
        canvas.bind('<Motion>', lambda event: self.move_pin_roi(event, transform=getattr(canvas, 'camera_transform', None)))
        canvas.bind('<Leave>', self.clear_pin_roi)
        # Keep the single-row controls reachable even when the view is resized.
        self.camera_popout.window.update_idletasks()
        toolbar_width = self.camera_popout.extra_controls.master.winfo_reqwidth()
        self.camera_popout.window.minsize(max(480, toolbar_width + 10), 320)

    def build_popout_tools(self, bar):
        # Shared variables keep both pages synchronized without extra setup dialogs.
        for label, mode in (('4 pins', 'Corners'), ('ROI', 'ROI')):
            tk.Radiobutton(bar, text=label, variable=self.mode, value=mode,
                           command=self.clear_pin_roi).pack(side='left', padx=1)
        tk.Checkbutton(bar, text='Manual', variable=self.manual_pin_var,
                       command=self.on_pin_method_changed).pack(side='left', padx=1)
        for label, variable in (('Pin r', self.pin_radius_var), ('ROI r', self.radius_var)):
            tk.Label(bar, text=label).pack(side='left', padx=(3, 1))
            tk.Entry(bar, textvariable=variable, width=4).pack(side='left')

    def set_command_running(self, running):
        self.command_running = running
        if self.camera_popout is not None and self.camera_popout.is_open:
            self.camera_popout.refresh()

    def click(self, event, *, transform=None):
        if getattr(self, 'command_running', False):
            return
        transform = self.transform if transform is None else transform
        if self.frame is None or transform is None:
            return
        if self.tracker is not None:
            self.status.config(text="Stop tracking before changing the calibration or ROI.")
            return
        ox, oy, scale = transform
        point = (round((event.x - ox) / scale), round((event.y - oy) / scale))
        h, w = self.frame.shape[:2]
        if not (0 <= point[0] < w and 0 <= point[1] < h):
            return
        if self.axis_busy:
            self.axis_status.config(text="Wait for the move to finish, then use Capture settled robot.")
            return
        if self.axis_pending is not None:
            self.axis_status.config(text="Use Capture settled robot, not a mouse click. Enlarge ROI radius if needed, or Reset axes to reposition the ROI.")
            return
        self.playing = False
        self.frozen = True
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
                self.axis_tracker = None
                self.axis_detections = []
                self.axis_processed_at = None
                self.status.config(text="ROI placed. Start tracking to detect robots and follow their center.")
            self.detections = []
            self.render()
        except ValueError as error:
            self.status.config(text=f"Not saved: {error}")
            self.render()
            messagebox.showerror("Calibration", str(error))

    def move_pin_roi(self, event, *, transform=None):
        if getattr(self, 'command_running', False):
            return
        transform = self.transform if transform is None else transform
        if (self.frame is None or transform is None or self.mode.get() != "Corners"
                or self.tracker is not None or self.axis_busy or self.axis_pending
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

    def start_tracking(self):
        if self.frame is None:
            messagebox.showerror("Calibration", "Connect the camera or open a movie first.")
            return
        try:
            if self.axis_busy or self.axis_pending or 0 < len(self.axis_samples) < 3:
                raise ValueError("Complete the robot-axis measurements or Reset axes before tracking.")
            if self.circle is None:
                raise ValueError("Place the ROI around the robots first.")
            circle = (*self.circle[:2], float(self.radius_var.get()))
            tracker = MovingROICalibration(
                self.corners, float(self.width_var.get()), float(self.height_var.get()),
                circle, int(self.count_var.get()), float(self.area_var.get()),
                getattr(self, "axis_samples", None) if len(getattr(self, "axis_samples", {})) == 3 else None,
                contrast_threshold=self.read_contrast_threshold(),
            )
            detections, center, _ = tracker.process(self.frame)
            if center is None:
                raise ValueError(
                    f"Found {len(detections)}/{tracker.count} robots. Enable Show detection mask: "
                    "the robot must appear white inside the ROI and four-point region. "
                    "Lower the dark contrast threshold (try 5), keep Min area at 1, and check Robots count."
                )
        except (ValueError, TypeError) as error:
            messagebox.showerror("Calibration", str(error))
            return
        self.clear_tracking()
        self.tracker = tracker
        self.axis_tracker = None
        self.axis_detections = []
        self.axis_processed_at = None
        self.live_started = time.monotonic()
        self.frozen = False
        self.accept_frame(self.frame)
        if self.capture is not None:
            self.playing = True
            if self.after_id is None:
                self.movie_tick()

    def stop_tracking(self):
        self.playing = False
        if self.tracker is not None:
            self.circle = (*self.tracker.center, self.tracker.radius)
        self.tracker = None
        self.status.config(text="Tracking stopped. Coordinates remain available for export.")

    def render(self, canvas=None):
        main_view = canvas is None
        canvas = self.canvas if main_view else canvas
        if self.frame is None:
            return
        annotated = self.frame.copy()
        h, w = annotated.shape[:2]
        cw, ch = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
        scale = min(cw / w, ch / h)

        def stroke(draw, *args, color, width=1, **kwargs):
            # One screen-pixel colour stroke; no extra black halo obscuring the scene.
            kwargs["line_type" if draw in (cv2.drawMarker, cv2.arrowedLine) else "lineType"] = cv2.LINE_AA
            draw(annotated, *args, color=color,
                 thickness=max(1, round(width / scale)), **kwargs)

        def marker(point, color):
            stroke(cv2.drawMarker, point, color=color, width=1,
                   markerType=cv2.MARKER_CROSS, markerSize=max(4, round(6 / scale)))

        def label(text, point, color):
            stroke(cv2.putText, text, point, color=color, width=1,
                   fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.32 / scale)

        mask_var = getattr(self, "show_mask_var", None)
        if mask_var is not None and mask_var.get():
            circle = self.circle if self.tracker is None else (*self.tracker.center, self.tracker.radius)
            mask = circle_detection_mask(self.frame.shape[:2], circle)
            if len(self.corners) == 4:
                workspace = workspace_detection_mask(self.frame.shape[:2], self.corners)
                mask = workspace if mask is None else cv2.bitwise_and(mask, workspace)
            try:
                _, candidates = self.detector.process(
                    self.frame, mode="Color blobs", color="Dark contrast",
                    minimum_area=float(self.area_var.get()), morphology_kernel_size=1,
                    draw_annotations=False, detection_mask=mask,
                    dark_contrast_threshold=self.read_contrast_threshold(),
                )
                candidates, _ = exclude_calibration_markers(candidates, self.corners, 10)
                annotated = cv2.cvtColor(self.detector.last_blob_mask, cv2.COLOR_GRAY2BGR)
                for item in candidates:
                    marker(item["center"], (0, 180, 255))
                self.detection_info.config(text=f"{len(candidates)} eligible blobs; white = contrast pixels")
            except (ValueError, TypeError):
                self.detection_info.config(text="Enter valid contrast and minimum area values.")
        elif hasattr(self, "detection_info"):
            self.detection_info.config(text="")
        pin_circle = getattr(self, "pin_search_circle", None)
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
            for name, color in (("+X", (0, 0, 255)), ("+Y", (255, 180, 0))):
                if name in samples:
                    point = tuple(map(int, samples[name]))
                    delta = np.asarray(point) - origin
                    length = np.linalg.norm(delta)
                    end = tuple(np.rint(np.asarray(origin) + delta / max(length, 1) * 45 / scale).astype(int))
                    stroke(cv2.arrowedLine, origin, end, color=color, tipLength=.15)
                    stroke(cv2.circle, point, max(3, round(3 / scale)), color=color, width=1)
                    label(name, end, color)
        dw, dh = max(1, round(w * scale)), max(1, round(h * scale))
        canvas.camera_transform = ((cw - dw) / 2, (ch - dh) / 2, scale)
        if main_view:
            self.transform = canvas.camera_transform
        image = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)).resize((dw, dh))
        canvas.camera_photo = ImageTk.PhotoImage(image)
        if main_view:
            self.photo = canvas.camera_photo
        canvas.delete("all")
        canvas.create_image(cw / 2, ch / 2, image=canvas.camera_photo, anchor="center")

    def setup_data(self):
        if getattr(self, "axis_busy", False) or getattr(self, "axis_pending", None):
            raise ValueError("Capture the current settled robot position before saving.")
        if self.frame is None or self.circle is None:
            raise ValueError("Select a frame, four points and an ROI first.")
        data = {
            "version": 2, "video_size_px": list(self.frame.shape[1::-1]),
            "corners_px": self.corners, "rectangle_width_cm": float(self.width_var.get()),
            "rectangle_height_cm": float(self.height_var.get()),
            "roi_circle_px": [
                *(self.circle[:2] if self.tracker is None else self.tracker.center),
                float(self.radius_var.get()) if self.tracker is None else self.tracker.radius,
            ],
            "robot_count": int(self.count_var.get()), "minimum_area": float(self.area_var.get()),
            "dark_contrast_threshold": self.read_contrast_threshold(),
            "axis_samples_px": getattr(self, "axis_samples", {}),
            "axis_targets_m": AXIS_TARGETS_M,
            "control_case": hardware_case_name(self.control_case_var.get()) if hasattr(self, "control_case_var") else "",
            "coordinate_system": "robot_axes" if len(getattr(self, "axis_samples", {})) == 3 else "rectangle_center",
        }
        if data["axis_samples_px"] and len(data["axis_samples_px"]) != 3:
            raise ValueError("Capture Center, +X and +Y or Reset axes before saving. Incomplete axes cannot be saved.")
        MovingROICalibration(data["corners_px"], data["rectangle_width_cm"],
                             data["rectangle_height_cm"], data["roi_circle_px"],
                             data["robot_count"], data["minimum_area"], data["axis_samples_px"],
                             contrast_threshold=data["dark_contrast_threshold"])
        return data

    def save_setup(self):
        try:
            data = self.setup_data()
            SETUP_DIR.mkdir(parents=True, exist_ok=True)
            path = filedialog.asksaveasfilename(
                title="Save GUI005 calibration", initialdir=str(SETUP_DIR),
                initialfile=default_setup_name(), defaultextension=".json",
                filetypes=[("Calibration setup", "*.json")],
            )
            if path:
                with open(path, "w", encoding="utf-8") as stream:
                    json.dump(data, stream, indent=2)
                self.status.config(text="Complete setup saved: four points, physical dimensions, ROI, detection settings, hardware case and XY calibration.")
        except (ValueError, OSError) as error:
            messagebox.showerror("Calibration", str(error))

    def load_setup(self):
        if self.frame is None:
            messagebox.showerror("Calibration", "Select a live frame or movie first.")
            return
        SETUP_DIR.mkdir(parents=True, exist_ok=True)
        path = filedialog.askopenfilename(
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
            samples = data.get("axis_samples_px", {})
            if samples and set(samples) != set(AXIS_TARGETS_M):
                raise ValueError("Saved robot axis references are incomplete.")
            tracker = MovingROICalibration(
                data["corners_px"], data["rectangle_width_cm"], data["rectangle_height_cm"],
                data["roi_circle_px"], data["robot_count"], data["minimum_area"],
                samples,
                contrast_threshold=data.get("dark_contrast_threshold", 10),
            )
            h, w = self.frame.shape[:2]
            points = np.asarray([*tracker.corners, tracker.center, *samples.values()], dtype=float)
            if np.any(points < 0) or np.any(points >= [w, h]):
                raise ValueError("Calibration points or ROI center are outside the frame.")
        except (ValueError, TypeError, KeyError, OSError) as error:
            messagebox.showerror("Calibration", str(error))
            return
        self.reset_axes()
        self.clear_tracking()
        self.corners, self.circle = tracker.corners, tuple(data["roi_circle_px"])
        self.pin_search_circle = None
        for variable, value in ((self.width_var, tracker.width_cm), (self.height_var, tracker.height_cm),
                                (self.radius_var, tracker.radius), (self.count_var, tracker.count),
                                (self.area_var, tracker.minimum_area)):
            variable.set(str(value))
        self.control_case_var.set(hardware_case_name(data.get("control_case", "")))
        if hasattr(self, "contrast_var"):
            self.contrast_var.set(str(tracker.contrast_threshold))
        self.axis_samples = samples
        self.axis_status.config(text="Saved robot origin and XY axes restored." if samples else "Setup has image-aligned XY. Calibrate robot axes if needed.")
        self.render()
        self.status.config(text="Setup loaded. Check four points and ROI, then Start tracking.")

    def export_csv(self):
        if not self.rows:
            messagebox.showerror("Coordinates", "Start tracking to collect coordinates first.")
            return
        path = filedialog.asksaveasfilename(title="Export robot XY", defaultextension=".csv")
        if path:
            try:
                with open(path, "w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["frame", "time_s", "group_valid", "robot_id", "x_px", "y_px",
                                     "x_cm", "y_cm", "group_x_cm", "group_y_cm", "roi_x_px", "roi_y_px"])
                    writer.writerows(self.rows)
                self.status.config(text="Robot and group XY coordinates exported.")
            except OSError as error:
                messagebox.showerror("Coordinates", str(error))

    def close(self):
        if getattr(self, 'camera_popout', None) is not None:
            self.camera_popout.close()
        self.axis_job += 1
        self.axis_busy = False
        if self.axis_after is not None:
            self.parent.after_cancel(self.axis_after)
            self.axis_after = None
        self.close_capture()
