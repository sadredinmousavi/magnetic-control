"""Offline video robot detection GUI."""

import csv
import json
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from robot_vision import RobotDetector


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_DIR / "inputs"
CALIBRATION_FILE = INPUT_DIR / "camera_calibration.json"
VIDEO_FILETYPES = [
    ("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v"),
    ("All files", "*.*"),
]


class OfflineDetectionGUI:
    """Process a recorded video as fast as possible and save its detections."""

    def __init__(self, root):
        self.root = root
        self.root.title("Offline Microrobot Detection")
        self.root.geometry("960x720")
        self.root.minsize(720, 540)

        self.detector = RobotDetector()
        self.stop_event = None
        self.worker_thread = None
        self.preview_queue = queue.Queue(maxsize=1)
        self.preview_photo = None
        self.preview_capture = None
        self.playback_after_id = None
        self.video_playing = False
        self.video_fps = 30.0
        self.video_total_frames = 0
        self.current_frame_number = 0
        self.current_frame_bgr = None
        self.display_transform = None
        self.calibration_points = []
        self.robot_points = []
        self.timeline_is_updating = False

        self.build_controls()
        self.build_viewer()
        self.root.after(50, self.poll_preview)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def build_controls(self):
        self.phase_notebook = ttk.Notebook(self.root)
        self.phase_notebook.pack(fill="x", padx=10, pady=(10, 4))
        self.phase_notebook.bind("<<NotebookTabChanged>>", self.on_phase_changed)

        calibration = tk.Frame(self.phase_notebook, padx=10, pady=8)
        robot_finding = tk.Frame(self.phase_notebook, padx=10, pady=8)
        processing = tk.Frame(self.phase_notebook, padx=10, pady=8)
        self.phase_notebook.add(calibration, text="Phase 1 - Calibration")
        self.phase_notebook.add(robot_finding, text="Phase 2 - Find robots")
        self.phase_notebook.add(processing, text="Phase 3 - Process")
        calibration.columnconfigure(1, weight=1)
        robot_finding.columnconfigure(1, weight=1)
        processing.columnconfigure(1, weight=1)

        tk.Label(calibration, text="Video file").grid(row=0, column=0, sticky="w")
        self.video_path_var = tk.StringVar()
        self.video_entry = tk.Entry(calibration, textvariable=self.video_path_var)
        self.video_entry.grid(
            row=0, column=1, columnspan=4, sticky="ew", padx=8, pady=4
        )
        self.browse_button = tk.Button(
            calibration, text="Browse...", command=self.choose_video
        )
        self.browse_button.grid(row=0, column=5, padx=4)

        self.play_button = tk.Button(
            calibration, text="Play", width=8, command=self.toggle_playback
        )
        self.play_button.grid(row=1, column=0, padx=4, pady=4)
        self.timeline = tk.Scale(
            calibration,
            from_=0,
            to=1,
            orient="horizontal",
            showvalue=False,
            command=self.on_timeline_changed,
        )
        self.timeline.grid(row=1, column=1, columnspan=4, sticky="ew", padx=8)
        self.time_label = tk.Label(calibration, text="00:00.000 / 00:00.000")
        self.time_label.grid(row=1, column=5, padx=4)

        tk.Label(calibration, text="h (cm)").grid(row=2, column=0, sticky="w")
        self.object_height_var = tk.StringVar(value="1")
        tk.Entry(
            calibration, textvariable=self.object_height_var, width=8
        ).grid(row=2, column=1, padx=8, pady=4, sticky="w")

        tk.Label(calibration, text="Camera to dot surface (cm)").grid(
            row=2, column=2, columnspan=2, sticky="e"
        )
        self.camera_height_var = tk.StringVar(value="29")
        tk.Entry(
            calibration, textvariable=self.camera_height_var, width=8
        ).grid(row=2, column=4, padx=8, pady=4, sticky="w")

        tk.Label(calibration, text="Dot rectangle width (cm)").grid(
            row=3, column=0, sticky="w"
        )
        self.rectangle_width_var = tk.StringVar(value="10")
        tk.Entry(
            calibration, textvariable=self.rectangle_width_var, width=8
        ).grid(row=3, column=1, padx=8, pady=4, sticky="w")

        tk.Label(calibration, text="Dot rectangle height (cm)").grid(
            row=3, column=2, columnspan=2, sticky="e"
        )
        self.rectangle_height_var = tk.StringVar(value="10")
        tk.Entry(
            calibration, textvariable=self.rectangle_height_var, width=8
        ).grid(row=3, column=4, padx=8, pady=4, sticky="w")

        tk.Label(calibration, text="Green-dot search radius (px)").grid(
            row=4, column=0, sticky="w"
        )
        self.calibration_search_radius_var = tk.StringVar(value="35")
        tk.Entry(
            calibration, textvariable=self.calibration_search_radius_var, width=8
        ).grid(row=4, column=1, padx=8, pady=4, sticky="w")

        self.calibration_status_label = tk.Label(
            calibration,
            text="Choose a video, find a clear frame, then click the 4 green dots.",
            fg="blue",
            anchor="w",
        )
        self.calibration_status_label.grid(
            row=5, column=0, columnspan=4, sticky="ew", pady=4
        )
        tk.Button(
            calibration, text="Clear points", command=self.clear_calibration_points
        ).grid(row=5, column=4, padx=4)
        tk.Button(
            calibration, text="Save calibration", command=self.save_calibration
        ).grid(row=5, column=5, padx=4)

        tk.Label(robot_finding, text="Finding method").grid(
            row=0, column=0, sticky="w"
        )
        self.robot_finding_mode_var = tk.StringVar(value="Manual selection")
        tk.OptionMenu(
            robot_finding,
            self.robot_finding_mode_var,
            "Manual selection",
            "Color detection",
        ).grid(row=0, column=1, sticky="w", padx=8, pady=4)

        tk.Label(robot_finding, text="Robot color").grid(
            row=0, column=2, sticky="e"
        )
        self.detection_color_var = tk.StringVar(value="Dark")
        tk.OptionMenu(
            robot_finding,
            self.detection_color_var,
            *RobotDetector.COLOR_RANGES,
        ).grid(row=0, column=3, sticky="w", padx=8, pady=4)

        tk.Label(robot_finding, text="Min area").grid(row=0, column=4, sticky="e")
        self.minimum_area_var = tk.StringVar(value="20")
        tk.Entry(
            robot_finding, textvariable=self.minimum_area_var, width=8
        ).grid(row=0, column=5, padx=4, pady=4)

        tk.Label(robot_finding, text="Number of robots").grid(
            row=1, column=0, sticky="w"
        )
        self.robot_count_var = tk.StringVar(value="3")
        tk.Entry(
            robot_finding, textvariable=self.robot_count_var, width=8
        ).grid(row=1, column=1, padx=8, pady=4, sticky="w")
        tk.Label(robot_finding, text="Click search radius (px)").grid(
            row=1, column=2, sticky="e"
        )
        self.robot_search_radius_var = tk.StringVar(value="35")
        tk.Entry(
            robot_finding, textvariable=self.robot_search_radius_var, width=8
        ).grid(row=1, column=3, padx=8, pady=4, sticky="w")
        tk.Button(
            robot_finding, text="Find by color", command=self.find_robots_by_color
        ).grid(row=1, column=4, padx=4)
        tk.Button(
            robot_finding, text="Clear robots", command=self.clear_robot_points
        ).grid(row=1, column=5, padx=4)
        self.robot_status_label = tk.Label(
            robot_finding,
            text="Choose manual selection and click each robot, or find them by color.",
            fg="blue",
            anchor="w",
        )
        self.robot_status_label.grid(
            row=2, column=0, columnspan=6, sticky="ew", pady=4
        )

        self.process_button = tk.Button(
            processing, text="Process Video", command=self.start_processing
        )
        self.process_button.grid(row=0, column=0, padx=4, pady=6, sticky="w")
        self.stop_button = tk.Button(
            processing, text="Stop", command=self.stop_processing, state="disabled"
        )
        self.stop_button.grid(row=0, column=1, padx=4, pady=6, sticky="w")

        self.progress = ttk.Progressbar(
            processing, orient="horizontal", mode="determinate", maximum=100
        )
        self.progress.grid(
            row=0, column=2, columnspan=4, sticky="ew", padx=8, pady=6
        )
        tk.Label(
            processing,
            text="Processes the full video using the Phase 1 area and Phase 2 color.",
            anchor="w",
        ).grid(row=1, column=0, columnspan=6, sticky="ew")

        self.status_label = tk.Label(
            self.root,
            text="Choose a recorded video, configure detection, and press Process Video.",
            fg="blue",
            anchor="w",
            padx=10,
        )
        self.status_label.pack(fill="x")
        self.detection_label = tk.Label(
            self.root, text="Detected robots: 0", fg="purple", anchor="w", padx=10
        )
        self.detection_label.pack(fill="x")

    def build_viewer(self):
        self.canvas = tk.Canvas(
            self.root, background="black", highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True, padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

    def choose_video(self):
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        selected = filedialog.askopenfilename(
            title="Select experimental video",
            initialdir=INPUT_DIR,
            filetypes=VIDEO_FILETYPES,
        )
        if selected:
            self.video_path_var.set(selected)
            self.open_calibration_video(Path(selected))

    def open_calibration_video(self, video_path):
        self.pause_playback()
        if self.preview_capture is not None:
            self.preview_capture.release()

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            messagebox.showerror("Video Error", f"Could not open video: {video_path}")
            self.preview_capture = None
            return

        self.preview_capture = capture
        self.video_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if self.video_fps <= 0:
            self.video_fps = 30.0
        self.video_total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.timeline.config(to=max(self.video_total_frames - 1, 1))
        self.calibration_points = []
        self.robot_points = []
        self.load_saved_calibration(width, height)
        self.show_calibration_frame(0)
        self.phase_notebook.select(0)

    def load_saved_calibration(self, video_width, video_height):
        if not CALIBRATION_FILE.is_file():
            return
        try:
            data = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
            saved_size = data.get("video_size_px", [])
            points = data.get("points_px", [])
            if saved_size != [video_width, video_height] or len(points) != 4:
                return
            self.calibration_points = [tuple(map(int, point)) for point in points]
            self.object_height_var.set(str(data.get("h_cm", 1)))
            self.camera_height_var.set(str(data.get("camera_height_cm", 29)))
            self.rectangle_width_var.set(str(data.get("rectangle_width_cm", 10)))
            self.rectangle_height_var.set(str(data.get("rectangle_height_cm", 10)))
            self.calibration_status_label.config(
                text="Loaded the saved 4-point calibration for this video size.",
                fg="green",
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            self.calibration_points = []

    def show_calibration_frame(self, frame_number, sequential=False):
        if self.preview_capture is None:
            return
        frame_number = max(0, min(int(frame_number), max(self.video_total_frames - 1, 0)))
        if not sequential:
            self.preview_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame_bgr = self.preview_capture.read()
        if not success:
            self.pause_playback()
            return

        self.current_frame_number = frame_number
        self.current_frame_bgr = frame_bgr
        self.timeline_is_updating = True
        self.timeline.set(frame_number)
        self.timeline_is_updating = False
        current_seconds = frame_number / self.video_fps
        total_seconds = (
            max(self.video_total_frames - 1, 0) / self.video_fps
            if self.video_total_frames
            else 0
        )
        self.time_label.config(
            text=f"{self.format_time(current_seconds)} / {self.format_time(total_seconds)}"
        )
        self.render_calibration_frame()

    @staticmethod
    def format_time(seconds):
        minutes, seconds = divmod(max(float(seconds), 0.0), 60.0)
        return f"{int(minutes):02d}:{seconds:06.3f}"

    def on_timeline_changed(self, value):
        if self.timeline_is_updating or self.preview_capture is None:
            return
        self.pause_playback()
        self.show_calibration_frame(round(float(value)))

    def toggle_playback(self):
        if self.preview_capture is None:
            video_path = Path(self.video_path_var.get().strip())
            if not video_path.is_file():
                messagebox.showerror("Video Error", "Select an existing video file.")
                return
            self.open_calibration_video(video_path)
        if self.video_playing:
            self.pause_playback()
            return
        if self.current_frame_number >= self.video_total_frames - 1:
            self.show_calibration_frame(0)
        self.video_playing = True
        self.play_button.config(text="Pause")
        self.play_next_frame()

    def play_next_frame(self):
        if not self.video_playing:
            return
        next_frame = self.current_frame_number + 1
        if next_frame >= self.video_total_frames:
            self.pause_playback()
            return
        self.show_calibration_frame(next_frame, sequential=True)
        delay_ms = max(1, round(1000.0 / self.video_fps))
        self.playback_after_id = self.root.after(delay_ms, self.play_next_frame)

    def pause_playback(self):
        self.video_playing = False
        if hasattr(self, "play_button"):
            self.play_button.config(text="Play")
        if self.playback_after_id is not None:
            self.root.after_cancel(self.playback_after_id)
            self.playback_after_id = None

    def on_canvas_click(self, event):
        phase = self.phase_notebook.index(self.phase_notebook.select())
        point = self.canvas_point_to_source(event.x, event.y)
        if point is None:
            return
        if phase == 0:
            self.add_calibration_point(point)
        elif phase == 1:
            self.add_robot_point(point)

    def canvas_point_to_source(self, canvas_x, canvas_y):
        if self.current_frame_bgr is None or self.display_transform is None:
            return None
        offset_x, offset_y, scale, source_width, source_height = self.display_transform
        source_x = round((canvas_x - offset_x) / scale)
        source_y = round((canvas_y - offset_y) / scale)
        if not (0 <= source_x < source_width and 0 <= source_y < source_height):
            return None
        return source_x, source_y

    def add_calibration_point(self, point):
        if len(self.calibration_points) >= 4:
            self.calibration_status_label.config(
                text="Four points are already selected. Clear them to select again.",
                fg="orange",
            )
            return

        snap_result = self.snap_calibration_point(point)
        if snap_result is None:
            return
        snapped_point, snapped, distance = snap_result
        self.calibration_points.append(snapped_point)
        if snapped:
            detail = f" Snapped {distance:.1f} px to the green-dot center."
            status_color = "blue"
        else:
            detail = " No nearby green dot found; kept the clicked position."
            status_color = "orange"
        if len(self.calibration_points) == 4:
            self.calibration_points = self.order_corner_points(self.calibration_points)
            message = "4/4 dots selected. Check the green quadrilateral, then save."
            if snapped:
                status_color = "green"
        else:
            message = f"{len(self.calibration_points)}/4 dots selected."
        self.calibration_status_label.config(
            text=message + detail, fg=status_color
        )
        self.render_calibration_frame()

    def snap_calibration_point(self, clicked_point):
        """Snap an approximate click to the nearest green calibration dot."""
        if self.current_frame_bgr is None:
            return clicked_point, False, 0.0
        try:
            radius = int(self.calibration_search_radius_var.get().strip())
            if radius <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Input Error", "Green-dot search radius must be positive."
            )
            return None
        return self.find_nearest_blob_center(
            clicked_point, color="Green", radius=radius, minimum_area=5.0
        )

    def add_robot_point(self, point):
        if self.robot_finding_mode_var.get() != "Manual selection":
            self.robot_status_label.config(
                text="Choose Manual selection to mark robots with the mouse.",
                fg="orange",
            )
            return
        robot_count = self.read_robot_count()
        if robot_count is None:
            return
        if len(self.robot_points) >= robot_count:
            self.robot_status_label.config(
                text=f"All {robot_count} robots are selected. Clear them to select again.",
                fg="orange",
            )
            return
        snap_result = self.snap_robot_point(point)
        if snap_result is None:
            return
        snapped_point, snapped, distance = snap_result
        self.robot_points.append(snapped_point)
        selected_count = len(self.robot_points)
        if snapped:
            detail = f"R{selected_count} snapped {distance:.1f} px to the blob center."
            status_color = "green" if selected_count == robot_count else "blue"
        else:
            detail = (
                f"R{selected_count}: no matching blob nearby; kept the clicked position."
            )
            status_color = "orange"
        self.robot_status_label.config(
            text=f"{selected_count}/{robot_count} robots selected. {detail}",
            fg=status_color,
        )
        self.render_calibration_frame()

    def snap_robot_point(self, clicked_point):
        """Snap an approximate click to the closest selected-color blob center."""
        if self.current_frame_bgr is None:
            return clicked_point, False, 0.0
        try:
            radius = int(self.robot_search_radius_var.get().strip())
            minimum_area = float(self.minimum_area_var.get().strip())
            if radius <= 0 or minimum_area <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Input Error", "Search radius and min area must be positive."
            )
            return None

        return self.find_nearest_blob_center(
            clicked_point,
            color=self.detection_color_var.get(),
            radius=radius,
            minimum_area=minimum_area,
        )

    def find_nearest_blob_center(self, clicked_point, color, radius, minimum_area):
        clicked_x, clicked_y = clicked_point
        frame_height, frame_width = self.current_frame_bgr.shape[:2]
        x0 = max(0, clicked_x - radius)
        y0 = max(0, clicked_y - radius)
        x1 = min(frame_width, clicked_x + radius + 1)
        y1 = min(frame_height, clicked_y + radius + 1)
        search_area = self.current_frame_bgr[y0:y1, x0:x1].copy()
        _, detections = self.detector.process(
            search_area,
            mode="Color blobs",
            color=color,
            minimum_area=minimum_area,
        )

        candidates = []
        for detection in detections:
            local_x, local_y = detection["center"]
            center = (local_x + x0, local_y + y0)
            distance = float(np.hypot(center[0] - clicked_x, center[1] - clicked_y))
            if distance <= radius:
                candidates.append((distance, center))
        if not candidates:
            return clicked_point, False, 0.0
        distance, center = min(candidates, key=lambda candidate: candidate[0])
        return center, True, distance

    def read_robot_count(self):
        try:
            robot_count = int(self.robot_count_var.get().strip())
            if robot_count <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Input Error", "Number of robots must be positive.")
            return None
        return robot_count

    def find_robots_by_color(self):
        if self.current_frame_bgr is None:
            messagebox.showerror("Video Error", "Choose a video first.")
            return
        if len(self.calibration_points) != 4:
            messagebox.showerror(
                "Calibration Error", "Select the four calibration dots first."
            )
            return
        robot_count = self.read_robot_count()
        if robot_count is None:
            return
        try:
            minimum_area = float(self.minimum_area_var.get().strip())
            if minimum_area <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Input Error", "Min area must be positive.")
            return

        height, width = self.current_frame_bgr.shape[:2]
        workspace_points = np.array(self.calibration_points, dtype=np.int32)
        workspace_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(workspace_mask, [workspace_points], 255)
        detection_frame = np.full_like(self.current_frame_bgr, 255)
        cv2.copyTo(self.current_frame_bgr, workspace_mask, detection_frame)
        _, detections = self.detector.process(
            detection_frame,
            mode="Color blobs",
            color=self.detection_color_var.get(),
            minimum_area=minimum_area,
        )
        self.robot_points = [item["center"] for item in detections[:robot_count]]
        self.robot_finding_mode_var.set("Color detection")
        found = len(self.robot_points)
        self.robot_status_label.config(
            text=f"Found {found}/{robot_count} robots by color.",
            fg="green" if found == robot_count else "orange",
        )
        self.render_calibration_frame()

    def clear_robot_points(self):
        self.robot_points = []
        self.robot_status_label.config(
            text="Robot selections cleared. Click robots or find them by color.",
            fg="blue",
        )
        self.render_calibration_frame()

    @staticmethod
    def order_corner_points(points):
        """Return top-left, top-right, bottom-right, bottom-left corner order."""
        top_left = min(points, key=lambda point: point[0] + point[1])
        bottom_right = max(points, key=lambda point: point[0] + point[1])
        top_right = max(points, key=lambda point: point[0] - point[1])
        bottom_left = min(points, key=lambda point: point[0] - point[1])
        return [top_left, top_right, bottom_right, bottom_left]

    def render_calibration_frame(self):
        if self.current_frame_bgr is None:
            return
        annotated = self.current_frame_bgr.copy()
        for index, point in enumerate(self.calibration_points, start=1):
            cv2.circle(annotated, point, 8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(
                annotated,
                f"P{index}",
                (point[0] + 10, point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        if len(self.calibration_points) >= 2:
            points = np.array(self.calibration_points, dtype=np.int32)
            cv2.polylines(
                annotated,
                [points],
                len(self.calibration_points) == 4,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        if self.phase_notebook.index(self.phase_notebook.select()) == 1:
            for index, point in enumerate(self.robot_points, start=1):
                cv2.circle(annotated, point, 10, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.drawMarker(
                    annotated, point, (0, 0, 255), cv2.MARKER_CROSS, 16, 2
                )
                cv2.putText(
                    annotated,
                    f"R{index}",
                    (point[0] + 12, point[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        self.show_preview(Image.fromarray(rgb))

    def clear_calibration_points(self):
        self.calibration_points = []
        self.calibration_status_label.config(
            text="Points cleared. Click the 4 green dots.", fg="blue"
        )
        self.render_calibration_frame()

    def save_calibration(self):
        if len(self.calibration_points) != 4 or self.current_frame_bgr is None:
            messagebox.showerror(
                "Calibration Error", "Select all four green dots before saving."
            )
            return
        geometry = self.read_geometry_values()
        if geometry is None:
            return
        object_height_cm, camera_height_cm, rectangle_width_cm, rectangle_height_cm = geometry
        height, width = self.current_frame_bgr.shape[:2]
        calibration = {
            "video_size_px": [width, height],
            "calibration_frame": self.current_frame_number,
            "points_px": [list(point) for point in self.calibration_points],
            "point_order": ["top_left", "top_right", "bottom_right", "bottom_left"],
            "h_cm": object_height_cm,
            "camera_height_cm": camera_height_cm,
            "rectangle_width_cm": rectangle_width_cm,
            "rectangle_height_cm": rectangle_height_cm,
        }
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        CALIBRATION_FILE.write_text(
            json.dumps(calibration, indent=2) + "\n", encoding="utf-8"
        )
        self.calibration_status_label.config(
            text=f"Calibration saved: {CALIBRATION_FILE}", fg="green"
        )

    def read_geometry_values(self):
        try:
            object_height_cm = float(self.object_height_var.get().strip())
            camera_height_cm = float(self.camera_height_var.get().strip())
            rectangle_width_cm = float(self.rectangle_width_var.get().strip())
            rectangle_height_cm = float(self.rectangle_height_var.get().strip())
            if (
                object_height_cm < 0
                or camera_height_cm <= 0
                or rectangle_width_cm <= 0
                or rectangle_height_cm <= 0
            ):
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Input Error",
                "h must be zero or positive; camera height and rectangle sizes "
                "must be positive.",
            )
            return None
        return (
            object_height_cm,
            camera_height_cm,
            rectangle_width_cm,
            rectangle_height_cm,
        )

    def on_phase_changed(self, _event=None):
        self.pause_playback()
        if self.phase_notebook.index(self.phase_notebook.select()) in (0, 1):
            self.render_calibration_frame()

    def on_canvas_resize(self, _event=None):
        if (
            hasattr(self, "phase_notebook")
            and self.phase_notebook.index(self.phase_notebook.select()) in (0, 1)
        ):
            self.render_calibration_frame()

    def start_processing(self):
        if self.worker_thread and self.worker_thread.is_alive():
            return

        video_path = Path(self.video_path_var.get().strip())
        if not video_path.is_file():
            messagebox.showerror("Video Error", "Select an existing video file.")
            return
        try:
            minimum_area = float(self.minimum_area_var.get().strip())
            if minimum_area <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Input Error", "Min area must be positive.")
            return
        geometry = self.read_geometry_values()
        if geometry is None:
            return
        (
            object_height_cm,
            camera_height_cm,
            rectangle_width_cm,
            rectangle_height_cm,
        ) = geometry
        if len(self.calibration_points) != 4:
            messagebox.showerror(
                "Calibration Error",
                "Complete Phase 1 by selecting the four green dots first.",
            )
            return
        robot_count = self.read_robot_count()
        if robot_count is None:
            return
        if len(self.robot_points) != robot_count:
            messagebox.showerror(
                "Robot Finding Error",
                f"Complete Phase 2 by selecting or finding all {robot_count} robots.",
            )
            return

        mode = "Color blobs"
        color = self.detection_color_var.get()
        self.pause_playback()
        self.stop_event = threading.Event()
        self.set_running(True)
        self.progress["value"] = 0
        self.status_label.config(text="Opening video...", fg="blue")

        self.worker_thread = threading.Thread(
            target=self.process_video,
            args=(
                video_path,
                mode,
                color,
                minimum_area,
                object_height_cm,
                camera_height_cm,
                rectangle_width_cm,
                rectangle_height_cm,
                list(self.calibration_points),
                self.stop_event,
            ),
            daemon=True,
        )
        self.worker_thread.start()

    def process_video(
        self,
        video_path,
        mode,
        color,
        minimum_area,
        object_height_cm,
        camera_height_cm,
        rectangle_width_cm,
        rectangle_height_cm,
        calibration_points,
        stop_event,
    ):
        capture = cv2.VideoCapture(str(video_path))
        writer = None
        csv_file = None
        result = None
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Could not open video: {video_path}")

            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if fps <= 0:
                fps = 30.0
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if width <= 0 or height <= 0:
                raise RuntimeError("The video does not report a valid frame size.")

            workspace_points = np.array(calibration_points, dtype=np.int32)
            workspace_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(workspace_mask, [workspace_points], 255)

            output_dir = PROJECT_DIR / "outputs" / "offline_detection" / video_path.stem
            output_dir.mkdir(parents=True, exist_ok=True)
            video_output = output_dir / f"{video_path.stem}_annotated.mp4"
            csv_output = output_dir / f"{video_path.stem}_detections.csv"

            writer = cv2.VideoWriter(
                str(video_output),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not create output video: {video_output}")

            csv_file = csv_output.open("w", newline="", encoding="utf-8")
            csv_writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "frame", "time_s", "h_cm", "camera_height_cm",
                    "dot_rectangle_width_cm", "dot_rectangle_height_cm",
                    "detection", "kind", "label",
                    "center_x_px", "center_y_px", "area_px2", "angle_deg",
                ],
            )
            csv_writer.writeheader()

            frame_number = 0
            while not stop_event.is_set():
                success, frame_bgr = capture.read()
                if not success:
                    break

                detection_frame = np.full_like(frame_bgr, 255)
                cv2.copyTo(frame_bgr, workspace_mask, detection_frame)
                annotated, detections = self.detector.process(
                    detection_frame,
                    mode=mode,
                    color=color,
                    minimum_area=minimum_area,
                )
                annotated[workspace_mask == 0] = frame_bgr[workspace_mask == 0]
                cv2.polylines(
                    annotated,
                    [workspace_points],
                    True,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                writer.write(annotated)
                time_s = frame_number / fps

                if detections:
                    for detection_number, detection in enumerate(detections, start=1):
                        center = detection.get("center", (None, None))
                        csv_writer.writerow({
                            "frame": frame_number,
                            "time_s": f"{time_s:.6f}",
                            "h_cm": object_height_cm,
                            "camera_height_cm": camera_height_cm,
                            "dot_rectangle_width_cm": rectangle_width_cm,
                            "dot_rectangle_height_cm": rectangle_height_cm,
                            "detection": detection_number,
                            "kind": detection.get("kind", ""),
                            "label": detection.get("label", ""),
                            "center_x_px": center[0],
                            "center_y_px": center[1],
                            "area_px2": detection.get("area", ""),
                            "angle_deg": detection.get("angle_deg", ""),
                        })
                else:
                    csv_writer.writerow({
                        "frame": frame_number,
                        "time_s": f"{time_s:.6f}",
                        "h_cm": object_height_cm,
                        "camera_height_cm": camera_height_cm,
                        "dot_rectangle_width_cm": rectangle_width_cm,
                        "dot_rectangle_height_cm": rectangle_height_cm,
                    })

                summary = self.detector.summarize(detections)
                progress = (
                    100.0 * (frame_number + 1) / total_frames
                    if total_frames > 0
                    else 0.0
                )
                if frame_number % 5 == 0:
                    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    self.put_latest_preview({
                        "kind": "frame",
                        "image": Image.fromarray(rgb),
                        "summary": summary,
                        "frame": frame_number + 1,
                        "total": total_frames,
                        "progress": progress,
                    })
                frame_number += 1

            stopped = stop_event.is_set()
            result = {
                "kind": "finished",
                "stopped": stopped,
                "frames": frame_number,
                "video_output": video_output,
                "csv_output": csv_output,
                "object_height_cm": object_height_cm,
                "camera_height_cm": camera_height_cm,
                "rectangle_width_cm": rectangle_width_cm,
                "rectangle_height_cm": rectangle_height_cm,
            }
        except Exception as exc:
            result = {"kind": "error", "message": str(exc)}
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            if csv_file is not None:
                csv_file.close()
        self.put_latest_preview(result)

    def put_latest_preview(self, item):
        try:
            self.preview_queue.put_nowait(item)
        except queue.Full:
            try:
                self.preview_queue.get_nowait()
            except queue.Empty:
                pass
            self.preview_queue.put_nowait(item)

    def poll_preview(self):
        item = None
        while not self.preview_queue.empty():
            try:
                item = self.preview_queue.get_nowait()
            except queue.Empty:
                break

        if item is not None:
            if item["kind"] == "frame":
                self.show_preview(item["image"])
                self.detection_label.config(text=item["summary"])
                self.progress["value"] = item["progress"]
                total = item["total"]
                total_text = str(total) if total > 0 else "?"
                self.status_label.config(
                    text=f"Processing frame {item['frame']}/{total_text}...",
                    fg="blue",
                )
            elif item["kind"] == "finished":
                self.set_running(False)
                self.progress["value"] = 100 if not item["stopped"] else self.progress["value"]
                state = "Stopped after" if item["stopped"] else "Completed"
                self.status_label.config(
                    text=(
                        f"{state} {item['frames']} frames. Video: "
                        f"{item['video_output']} | CSV: {item['csv_output']}"
                    ),
                    fg="orange" if item["stopped"] else "green",
                )
            elif item["kind"] == "error":
                self.set_running(False)
                self.status_label.config(text=f"Processing failed: {item['message']}", fg="red")

        self.root.after(50, self.poll_preview)

    def show_preview(self, image):
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)
        source_width, source_height = image.size
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        displayed_width, displayed_height = image.size
        offset_x = (width - displayed_width) / 2.0
        offset_y = (height - displayed_height) / 2.0
        scale = displayed_width / source_width
        self.display_transform = (
            offset_x,
            offset_y,
            scale,
            source_width,
            source_height,
        )
        self.preview_photo = ImageTk.PhotoImage(image)
        self.canvas.delete("video_frame")
        self.canvas.create_image(
            width // 2,
            height // 2,
            image=self.preview_photo,
            anchor="center",
            tags="video_frame",
        )

    def set_running(self, running):
        self.process_button.config(state="disabled" if running else "normal")
        self.browse_button.config(state="disabled" if running else "normal")
        self.video_entry.config(state="disabled" if running else "normal")
        self.stop_button.config(state="normal" if running else "disabled")
        self.phase_notebook.tab(0, state="disabled" if running else "normal")
        self.phase_notebook.tab(1, state="disabled" if running else "normal")

    def stop_processing(self):
        if self.stop_event is not None:
            self.stop_event.set()
            self.status_label.config(text="Stop requested...", fg="orange")

    def close(self):
        self.pause_playback()
        if self.preview_capture is not None:
            self.preview_capture.release()
        if self.stop_event is not None:
            self.stop_event.set()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = OfflineDetectionGUI(root)
    root.mainloop()
