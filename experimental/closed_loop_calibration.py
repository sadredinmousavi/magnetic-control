"""GUI005 calibration using GUI004's dark-contrast moving swarm ROI."""

import csv
import json
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from gui004 import (
    OfflineDetectionGUI, calibrated_dwell_coordinates, circle_detection_mask,
    exclude_calibration_markers, match_compact_swarm_candidates,
    robot_center_for, workspace_detection_mask,
)
from robot_vision import RobotDetector


class MovingROICalibration:
    """Frame-by-frame tracking shared by live preview and movie playback."""

    def __init__(self, corners, width_cm, height_cm, circle, count, minimum_area):
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
        self.detector = RobotDetector()

    def process(self, frame):
        mask = circle_detection_mask(frame.shape[:2], (*self.center, self.radius))
        mask = cv2.bitwise_and(mask, workspace_detection_mask(frame.shape[:2], self.corners))
        _, candidates = self.detector.process(
            frame, mode="Color blobs", color="Dark contrast",
            minimum_area=self.minimum_area, morphology_kernel_size=1,
            draw_annotations=False, detection_mask=mask,
        )
        candidates, _ = exclude_calibration_markers(candidates, self.corners, 10)
        if self.references:
            selected = match_compact_swarm_candidates(
                candidates, self.references, self.count, self.radius,
                predicted_center=self.center, swarm_radius=self.radius,
            )
        else:
            selected = [dict(item, robot_id=index + 1)
                        for index, item in enumerate(candidates[:self.count])]
        center = robot_center_for(selected, self.count)
        # Only a complete group moves the ROI or produces a valid group XY.
        # Missing robots leave a gap rather than shifting toward a partial group.
        if center is not None:
            self.center = np.asarray(center, dtype=float)
            self.references = [item["center"] for item in sorted(
                selected, key=lambda item: item["robot_id"])]
        xy = calibrated_dwell_coordinates(
            [{"center": item["center"]} for item in selected],
            self.corners, self.width_cm, self.height_cm,
        )
        for item, coordinates in zip(selected, xy):
            item["xy_cm"] = tuple(coordinates)
        group_xy = None if center is None else tuple(calibrated_dwell_coordinates(
            [{"center": center}], self.corners, self.width_cm, self.height_cm,
        )[0])
        return selected, center, group_xy


class CalibrationTab:
    def __init__(self, parent):
        self.parent = parent
        self.frame = None
        self.corners = []
        self.circle = None
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

        bar = tk.Frame(parent)
        bar.pack(fill="x", padx=8, pady=6)
        for label, command in (
            ("Live camera", self.use_live), ("Open movie", self.open_movie),
            ("Play / Pause", self.toggle_play), ("First frame", self.rewind),
            ("Save setup", self.save_setup), ("Load setup", self.load_setup),
            ("Export XY CSV", self.export_csv),
        ):
            tk.Button(bar, text=label, command=command).pack(side="left", padx=2)
        settings = tk.Frame(parent)
        settings.pack(fill="x", padx=8, pady=3)
        self.width_var = tk.StringVar(value="10")
        self.height_var = tk.StringVar(value="10")
        self.radius_var = tk.StringVar(value="60")
        self.count_var = tk.StringVar(value="3")
        self.area_var = tk.StringVar(value="1")
        for label, variable in (("Width (cm)", self.width_var),
                                ("Height (cm)", self.height_var),
                                ("ROI radius (px)", self.radius_var),
                                ("Robots", self.count_var),
                                ("Min area (px²)", self.area_var)):
            tk.Label(settings, text=label).pack(side="left", padx=(4, 2))
            tk.Entry(settings, textvariable=variable, width=6).pack(side="left")
        tools = tk.Frame(parent)
        tools.pack(fill="x", padx=8, pady=4)
        self.mode = tk.StringVar(value="ROI")
        for label, value in (("Select 4 green points", "Corners"), ("Place ROI center", "ROI")):
            ttk.Radiobutton(tools, text=label, value=value, variable=self.mode).pack(side="left", padx=4)
        tk.Button(tools, text="Clear points", command=self.clear_points).pack(side="left", padx=4)
        tk.Button(tools, text="Freeze / Resume live", command=self.freeze_live).pack(side="left", padx=4)
        tk.Button(tools, text="Start tracking", command=self.start_tracking).pack(side="left", padx=4)
        tk.Button(tools, text="Stop tracking", command=self.stop_tracking).pack(side="left", padx=4)
        tk.Label(parent, anchor="w", text=(
            "Select four corner dots, enter their rectangle size, then place a circular ROI around all robots. "
            "Dark contrast detection only. XY origin: rectangle center; X right, Y up."
        ), wraplength=1000).pack(fill="x", padx=10)
        self.status = tk.Label(parent, text="Waiting for a live camera frame or movie.", anchor="w")
        self.status.pack(fill="x", padx=10, pady=4)
        self.canvas = tk.Canvas(parent, background="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.canvas.bind("<Button-1>", self.click)
        self.canvas.bind("<Configure>", lambda event: self.render())

    def clear_tracking(self):
        self.tracker = None
        self.detections = []
        self.rows = []
        self.live_started = None

    def close_capture(self):
        self.playing = False
        if self.after_id is not None:
            self.parent.after_cancel(self.after_id)
            self.after_id = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def use_live(self):
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
            self.clear_tracking()
            self.status.config(text="Frame size changed. Select four points and place the ROI again.")
        self.frame = frame.copy()
        if self.tracker is not None:
            try:
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
        if self.capture is None:
            self.frozen = not self.frozen
            self.status.config(text="Live frame frozen for setup." if self.frozen else "Live preview resumed.")

    def click(self, event):
        if self.frame is None or self.transform is None:
            return
        if self.tracker is not None:
            self.status.config(text="Stop tracking before changing the calibration or ROI.")
            return
        ox, oy, scale = self.transform
        point = (round((event.x - ox) / scale), round((event.y - oy) / scale))
        h, w = self.frame.shape[:2]
        if not (0 <= point[0] < w and 0 <= point[1] < h):
            return
        self.playing = False
        self.frozen = True
        try:
            if self.mode.get() == "Corners":
                if len(self.corners) == 4:
                    self.status.config(text="Four points selected. Clear points to select again.")
                    return
                mask = circle_detection_mask((h, w), (*point, 35))
                _, dots = self.detector.process(
                    self.frame, mode="Color blobs", color="Green", minimum_area=1,
                    morphology_kernel_size=1, draw_annotations=False, detection_mask=mask,
                )
                snapped = bool(dots)
                if dots:
                    point = min(dots, key=lambda item: np.linalg.norm(
                        np.asarray(item["center"]) - point))["center"]
                self.corners.append(point)
                if len(self.corners) == 4:
                    self.corners = OfflineDetectionGUI.order_corner_points(self.corners)
                self.status.config(text=f"{len(self.corners)}/4 points. " + (
                    "Snapped to green dot." if snapped else "No nearby green dot; kept clicked position."))
            else:
                radius = float(self.radius_var.get())
                if not np.isfinite(radius) or radius <= 0:
                    raise ValueError("ROI radius must be positive.")
                self.circle = (*point, radius)
                self.status.config(text="ROI placed. Start tracking to detect robots and follow their center.")
            self.detections = []
            self.render()
        except ValueError as error:
            messagebox.showerror("Calibration", str(error))

    def clear_points(self):
        self.stop_tracking()
        self.corners = []
        self.render()
        self.status.config(text="Select the four green corner dots.")

    def start_tracking(self):
        if self.frame is None:
            messagebox.showerror("Calibration", "Connect the camera or open a movie first.")
            return
        try:
            if self.circle is None:
                raise ValueError("Place the ROI around the robots first.")
            circle = (*self.circle[:2], float(self.radius_var.get()))
            tracker = MovingROICalibration(
                self.corners, float(self.width_var.get()), float(self.height_var.get()),
                circle, int(self.count_var.get()), float(self.area_var.get()),
            )
            detections, center, _ = tracker.process(self.frame)
            if center is None:
                raise ValueError(f"Found {len(detections)}/{tracker.count} robots. Adjust ROI, robot count or min area.")
        except (ValueError, TypeError) as error:
            messagebox.showerror("Calibration", str(error))
            return
        self.clear_tracking()
        self.tracker = tracker
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

    def render(self):
        if self.frame is None:
            return
        annotated = self.frame.copy()
        if len(self.corners) == 4:
            cv2.polylines(annotated, [np.asarray(self.corners, np.int32)], True, (0, 255, 0), 1)
        for index, point in enumerate(self.corners, 1):
            p = tuple(map(int, point))
            cv2.drawMarker(annotated, p, (0, 255, 0), cv2.MARKER_CROSS, 12, 1)
            cv2.putText(annotated, f"P{index}", (p[0] + 5, p[1] - 5), 0, .5, (0, 255, 0), 1)
        circle = self.circle if self.tracker is None else (*self.tracker.center, self.tracker.radius)
        if circle is not None:
            cv2.circle(annotated, tuple(map(lambda v: round(float(v)), circle[:2])),
                       round(float(circle[2])), (255, 255, 0), 1)
        for item in self.detections:
            cv2.drawMarker(annotated, item["center"], (0, 0, 255), cv2.MARKER_CROSS, 10, 1)
        h, w = annotated.shape[:2]
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        scale = min(cw / w, ch / h)
        dw, dh = max(1, round(w * scale)), max(1, round(h * scale))
        self.transform = ((cw - dw) / 2, (ch - dh) / 2, scale)
        image = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)).resize((dw, dh))
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(cw / 2, ch / 2, image=self.photo, anchor="center")

    def setup_data(self):
        if self.frame is None or self.circle is None:
            raise ValueError("Select a frame, four points and an ROI first.")
        data = {
            "version": 1, "video_size_px": list(self.frame.shape[1::-1]),
            "corners_px": self.corners, "rectangle_width_cm": float(self.width_var.get()),
            "rectangle_height_cm": float(self.height_var.get()),
            "roi_circle_px": [*self.circle[:2], float(self.radius_var.get())],
            "robot_count": int(self.count_var.get()), "minimum_area": float(self.area_var.get()),
        }
        MovingROICalibration(data["corners_px"], data["rectangle_width_cm"],
                             data["rectangle_height_cm"], data["roi_circle_px"],
                             data["robot_count"], data["minimum_area"])
        return data

    def save_setup(self):
        try:
            data = self.setup_data()
            path = filedialog.asksaveasfilename(title="Save GUI005 calibration", defaultextension=".json")
            if path:
                with open(path, "w", encoding="utf-8") as stream:
                    json.dump(data, stream, indent=2)
                self.status.config(text="Calibration setup saved.")
        except (ValueError, OSError) as error:
            messagebox.showerror("Calibration", str(error))

    def load_setup(self):
        if self.frame is None:
            messagebox.showerror("Calibration", "Select a live frame or movie first.")
            return
        path = filedialog.askopenfilename(title="Load GUI005 calibration", filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as stream:
                data = json.load(stream)
            if data["version"] != 1 or data["video_size_px"] != list(self.frame.shape[1::-1]):
                raise ValueError("Calibration does not match the current frame size.")
            tracker = MovingROICalibration(
                data["corners_px"], data["rectangle_width_cm"], data["rectangle_height_cm"],
                data["roi_circle_px"], data["robot_count"], data["minimum_area"],
            )
            h, w = self.frame.shape[:2]
            points = np.asarray([*tracker.corners, tracker.center], dtype=float)
            if np.any(points < 0) or np.any(points >= [w, h]):
                raise ValueError("Calibration points or ROI center are outside the frame.")
        except (ValueError, TypeError, KeyError, OSError) as error:
            messagebox.showerror("Calibration", str(error))
            return
        self.stop_tracking()
        self.clear_tracking()
        self.corners, self.circle = tracker.corners, tuple(data["roi_circle_px"])
        for variable, value in ((self.width_var, tracker.width_cm), (self.height_var, tracker.height_cm),
                                (self.radius_var, tracker.radius), (self.count_var, tracker.count),
                                (self.area_var, tracker.minimum_area)):
            variable.set(str(value))
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
        self.close_capture()
