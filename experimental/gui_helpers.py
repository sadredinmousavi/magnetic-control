import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None

try:
    import cv2

    from robot_vision import RobotDetector
except ImportError:
    cv2 = None
    RobotDetector = None

try:
    from cv2_enumerate_cameras import enumerate_cameras
except ImportError:
    enumerate_cameras = None

try:
    import serial
except ImportError:
    serial = None


class BaseServoGUI:
    CAMERA_BACKENDS = {
        "Auto": None,
        "Media Foundation": "CAP_MSMF",
        "DirectShow": "CAP_DSHOW",
        "Default": None,
    }

    def __init__(self, root, title="Servo GUI", geometry="800x600"):
        self.root = root
        self.root.title(title)
        self.root.geometry(geometry)

        self.port = None
        self.camera_window = None
        self.camera_canvas = None
        self.camera_status_label = None
        self.camera_capture = None
        self.camera_stop = None
        self.camera_frames = queue.Queue(maxsize=1)
        self.camera_photo = None
        self.camera_detection_label = None
        self.camera_detection_mode = "Off"
        self.camera_detection_color = "Red"
        self.camera_detection_minimum_area = 150.0
        self.camera_index = 0
        self.camera_backend = "Auto"
        self.camera_choices = {}
        self.camera_scan_thread = None
        self.robot_detector = RobotDetector() if RobotDetector is not None else None

    def maximize_window(self):
        try:
            self.root.state("zoomed")
        except tk.TclError:
            self.root.attributes("-zoomed", True)

    # Linux default: /dev/ttyS0
    def build_connection(
        self,
        default_port="COM5",
        default_baud="1000000",
        show_disconnect=False,
        show_camera=False,
    ):
        frame = tk.LabelFrame(self.root, text="Connection", padx=10, pady=10)
        frame.pack(fill="x", padx=10, pady=5)

        tk.Label(frame, text="Port").grid(row=0, column=0, sticky="w")
        self.port_entry = tk.Entry(frame, width=20)
        self.port_entry.insert(0, default_port)
        self.port_entry.grid(row=0, column=1, padx=5)

        tk.Label(frame, text="Baudrate").grid(row=0, column=2, sticky="w")
        self.baud_entry = tk.Entry(frame, width=15)
        self.baud_entry.insert(0, default_baud)
        self.baud_entry.grid(row=0, column=3, padx=5)

        tk.Button(frame, text="Connect", command=self.connect_port).grid(
            row=0, column=4, padx=5
        )

        if show_disconnect:
            tk.Button(frame, text="Close Connection", command=self.disconnect_port).grid(
                row=0, column=5, padx=5
            )

        if show_camera:
            tk.Button(frame, text="Webcam", command=self.open_camera_window).grid(
                row=0, column=6, padx=5
            )

        return frame

    def build_status(self):
        frame = tk.LabelFrame(self.root, text="Status / Log", padx=10, pady=10)
        frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.status_label = tk.Label(frame, text="Not connected", fg="red")
        self.status_label.pack(anchor="w")

        self.log_text = tk.Text(frame, height=10)
        self.log_text.pack(fill="both", expand=True)

        return frame

    def log(self, text):
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)

    def set_status(self, text, color="black"):
        self.status_label.config(text=text, fg=color)
        self.log(text)

    def connect_port(self):
        if serial is None:
            messagebox.showerror("Error", "pyserial is not installed")
            return

        try:
            if self.port and getattr(self.port, "is_open", False):
                self.port.close()

            port_name = self.port_entry.get().strip()
            baudrate = int(self.baud_entry.get().strip())

            self.port = serial.Serial(port_name, baudrate=baudrate)
            self.set_status(f"Connected to {port_name}", "green")

        except Exception as e:
            self.set_status("Connection failed", "red")
            messagebox.showerror("Connection Error", str(e))

    def disconnect_port(self):
        try:
            if self.port and getattr(self.port, "is_open", False):
                self.port.close()
            self.port = None
            self.set_status("Serial connection closed", "red")
        except Exception as e:
            self.set_status("Could not close serial connection", "red")
            messagebox.showerror("Connection Error", str(e))

    def open_camera_window(self):
        if Image is None or ImageTk is None:
            messagebox.showerror(
                "Camera Error",
                "Pillow is not installed. Install the experimental requirements first.",
            )
            return

        if self.camera_window and self.camera_window.winfo_exists():
            self.camera_window.deiconify()
            self.camera_window.lift()
            self.camera_window.focus_force()
            return

        self.camera_window = tk.Toplevel(self.root)
        self.camera_window.title("Webcam Viewer")
        self.camera_window.geometry("960x720")
        self.camera_window.minsize(640, 480)
        self.camera_window.protocol("WM_DELETE_WINDOW", self.close_camera_window)

        controls = tk.Frame(self.camera_window, padx=10, pady=10)
        controls.pack(fill="x")
        controls.columnconfigure(1, weight=1)

        tk.Label(controls, text="Camera").grid(row=0, column=0, sticky="w")
        self.camera_choice_var = tk.StringVar(value="Scanning for cameras...")
        self.camera_choice_menu = tk.OptionMenu(
            controls,
            self.camera_choice_var,
            "Scanning for cameras...",
        )
        self.camera_choice_menu.grid(
            row=0, column=1, columnspan=3, sticky="ew", padx=8, pady=4
        )

        self.camera_scan_button = tk.Button(
            controls, text="Rescan", command=self.scan_cameras
        )
        self.camera_scan_button.grid(row=0, column=4, padx=4)

        self.camera_connect_button = tk.Button(
            controls, text="Connect", command=self.connect_camera, state="disabled"
        )
        self.camera_connect_button.grid(row=0, column=5, padx=4)
        tk.Button(controls, text="Disconnect", command=self.disconnect_camera).grid(
            row=0, column=6, padx=4
        )

        tk.Label(controls, text="Detection").grid(row=1, column=0, sticky="w")
        self.camera_detection_mode_var = tk.StringVar(
            value=self.camera_detection_mode
        )
        detection_menu = tk.OptionMenu(
            controls,
            self.camera_detection_mode_var,
            "Off",
            "Color blobs",
            "ArUco markers",
            "Color + ArUco",
            command=lambda _value: self.update_camera_detection_settings(),
        )
        detection_menu.grid(row=1, column=1, sticky="w", padx=8, pady=4)

        tk.Label(controls, text="Target color").grid(row=1, column=2, sticky="e")
        self.camera_detection_color_var = tk.StringVar(
            value=self.camera_detection_color
        )
        color_menu = tk.OptionMenu(
            controls,
            self.camera_detection_color_var,
            *RobotDetector.COLOR_RANGES.keys() if RobotDetector is not None else ["Red"],
            command=lambda _value: self.update_camera_detection_settings(),
        )
        color_menu.grid(row=1, column=3, sticky="w", padx=8, pady=4)

        tk.Label(controls, text="Min area").grid(row=1, column=4, sticky="e")
        self.camera_detection_area_entry = tk.Entry(controls, width=8)
        self.camera_detection_area_entry.insert(
            0, str(int(self.camera_detection_minimum_area))
        )
        self.camera_detection_area_entry.grid(row=1, column=5, padx=4, pady=4)
        self.camera_detection_area_entry.bind(
            "<KeyRelease>", lambda _event: self.update_camera_detection_settings()
        )

        if self.robot_detector is None:
            detection_menu.config(state="disabled")

        self.camera_status_label = tk.Label(
            self.camera_window,
            text="Scanning for connected cameras...",
            fg="blue",
            anchor="w",
            padx=10,
        )
        self.camera_status_label.pack(fill="x")

        self.camera_detection_label = tk.Label(
            self.camera_window,
            text=(
                "Detected robots: 0"
                if self.robot_detector is not None
                else "OpenCV is not installed; detection is unavailable."
            ),
            fg="purple",
            anchor="w",
            padx=10,
        )
        self.camera_detection_label.pack(fill="x")

        self.camera_canvas = tk.Canvas(
            self.camera_window, background="black", highlightthickness=0
        )
        self.camera_canvas.pack(fill="both", expand=True, padx=10, pady=10)
        self.camera_window.after(50, self.poll_camera_frames)
        self.scan_cameras()

    def update_camera_detection_settings(self):
        self.camera_detection_mode = self.camera_detection_mode_var.get()
        self.camera_detection_color = self.camera_detection_color_var.get()
        try:
            value = float(self.camera_detection_area_entry.get().strip())
            if value > 0:
                self.camera_detection_minimum_area = value
        except ValueError:
            pass

    def connect_camera(self):
        if cv2 is None:
            messagebox.showerror(
                "Camera Error",
                "OpenCV is not installed. Install the experimental requirements first.",
            )
            return
        selected = self.camera_choice_var.get()
        if selected not in self.camera_choices:
            messagebox.showerror(
                "Camera Error", "Select a detected camera or press Rescan."
            )
            return
        camera_index, camera_backend = self.camera_choices[selected]
        self.camera_index = camera_index
        self.camera_backend = camera_backend

        self.disconnect_camera(update_status=False)
        while not self.camera_frames.empty():
            try:
                self.camera_frames.get_nowait()
            except queue.Empty:
                break

        stop_event = threading.Event()
        self.camera_stop = stop_event
        self.set_camera_status("Connecting to camera...", "blue")
        threading.Thread(
            target=self.camera_read_loop,
            args=(camera_index, camera_backend, stop_event),
            daemon=True,
        ).start()

    @staticmethod
    def read_camera_frame_with_warmup(capture, attempts=20):
        """Allow USB cameras time to negotiate a format and produce a frame."""
        for _ in range(attempts):
            success, frame = capture.read()
            if success and frame is not None:
                return frame
            time.sleep(0.05)

        # Some USB UVC cameras fail their default uncompressed format but work
        # when a common MJPEG 640x480 mode is requested explicitly.
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        capture.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG"),
        )
        for _ in range(attempts):
            success, frame = capture.read()
            if success and frame is not None:
                return frame
            time.sleep(0.05)
        return None

    @staticmethod
    def discover_cameras(max_index=15):
        """Enumerate named cameras, with active probing as a fallback."""
        if cv2 is None:
            return []

        if enumerate_cameras is not None:
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            camera_infos = enumerate_cameras(backend)
            return [
                {
                    "index": int(camera.index),
                    "backend": "Auto" if sys.platform == "win32" else "Default",
                    "name": camera.name,
                    "vid": camera.vid,
                    "pid": camera.pid,
                    "width": None,
                    "height": None,
                }
                for camera in camera_infos
            ]

        if sys.platform == "win32":
            backend_specs = (
                ("Media Foundation", "CAP_MSMF"),
                ("DirectShow", "CAP_DSHOW"),
                ("Default", None),
            )
        else:
            backend_specs = (("Default", None),)

        cameras = []
        for backend_name, attribute_name in backend_specs:
            backend = getattr(cv2, attribute_name, None) if attribute_name else None
            if attribute_name and backend is None:
                continue
            for camera_index in range(max_index + 1):
                capture = None
                try:
                    capture = (
                        cv2.VideoCapture(camera_index, backend)
                        if backend is not None
                        else cv2.VideoCapture(camera_index)
                    )
                    if not capture.isOpened():
                        continue
                    frame = BaseServoGUI.read_camera_frame_with_warmup(capture)
                    if frame is None:
                        continue
                    height, width = frame.shape[:2]
                    cameras.append({
                        "index": camera_index,
                        "backend": backend_name,
                        "width": int(width),
                        "height": int(height),
                    })
                except Exception:
                    # A backend can be installed but unavailable on a particular
                    # PC. Continue probing the remaining backends and indices.
                    continue
                finally:
                    if capture is not None:
                        capture.release()

        # OpenCV's Default backend normally delegates to one of the explicit
        # Windows backends. Hide only those exact Default duplicates while
        # retaining distinct Media Foundation and DirectShow choices.
        explicit = {
            (camera["index"], camera["width"], camera["height"])
            for camera in cameras
            if camera["backend"] != "Default"
        }
        return [
            camera
            for camera in cameras
            if camera["backend"] != "Default"
            or (camera["index"], camera["width"], camera["height"]) not in explicit
        ]

    def scan_cameras(self):
        """Discover connected cameras without blocking the Tk event loop."""
        if cv2 is None:
            messagebox.showerror(
                "Camera Error",
                "OpenCV is not installed. Install the experimental requirements first.",
            )
            return
        if self.camera_scan_thread and self.camera_scan_thread.is_alive():
            return

        self.disconnect_camera(update_status=False)
        self.camera_choices = {}
        self.camera_choice_var.set("Scanning for cameras...")
        self.camera_choice_menu.config(state="disabled")
        self.camera_connect_button.config(state="disabled")
        self.camera_scan_button.config(state="disabled")
        self.set_camera_status("Scanning camera indices 0-15...", "blue")

        def worker():
            try:
                cameras = self.discover_cameras()
                error = None
            except Exception as exc:
                cameras = []
                error = exc
            self.root.after(0, lambda: self.finish_camera_scan(cameras, error))

        self.camera_scan_thread = threading.Thread(target=worker, daemon=True)
        self.camera_scan_thread.start()

    def finish_camera_scan(self, cameras, error=None):
        """Populate the camera dropdown after background discovery finishes."""
        self.camera_scan_thread = None
        if not self.camera_window or not self.camera_window.winfo_exists():
            return

        menu = self.camera_choice_menu["menu"]
        menu.delete(0, "end")
        self.camera_scan_button.config(state="normal")

        if error is not None:
            self.camera_choice_var.set("Camera scan failed")
            self.set_camera_status(f"Camera scan failed: {error}", "red")
            return
        if not cameras:
            self.camera_choice_var.set("No cameras found")
            self.set_camera_status(
                "No working cameras found. Check the USB connection, camera privacy "
                "permissions, and other applications, then press Rescan.",
                "red",
            )
            return

        for camera in cameras:
            name = camera.get("name") or f"Camera {camera['index']}"
            device_id = ""
            if camera.get("vid") is not None and camera.get("pid") is not None:
                device_id = f" [{camera['vid']:04X}:{camera['pid']:04X}]"
            resolution = ""
            if camera.get("width") is not None and camera.get("height") is not None:
                resolution = f" ({camera['width']}x{camera['height']})"
            label = (
                f"{name}{device_id} — index {camera['index']}"
                f" / {camera['backend']}{resolution}"
            )
            self.camera_choices[label] = (camera["index"], camera["backend"])
            menu.add_command(
                label=label,
                command=tk._setit(self.camera_choice_var, label),
            )

        first_label = next(iter(self.camera_choices))
        self.camera_choice_var.set(first_label)
        self.camera_choice_menu.config(state="normal")
        self.camera_connect_button.config(state="normal")
        self.set_camera_status(
            f"Found {len(cameras)} camera device(s).", "green"
        )

    @staticmethod
    def open_camera_capture(camera_index, camera_backend):
        """Open a camera with the selected backend or an ordered fallback list."""
        if camera_backend == "Auto":
            backend_names = (
                ("Media Foundation", "CAP_MSMF"),
                ("DirectShow", "CAP_DSHOW"),
                ("Default", None),
            )
        else:
            backend_names = ((camera_backend, BaseServoGUI.CAMERA_BACKENDS[camera_backend]),)

        attempts = []
        for backend_name, attribute_name in backend_names:
            backend = getattr(cv2, attribute_name) if attribute_name else None
            capture = (
                cv2.VideoCapture(camera_index, backend)
                if backend is not None
                else cv2.VideoCapture(camera_index)
            )
            if capture.isOpened():
                return capture, backend_name
            capture.release()
            attempts.append(backend_name)

        attempted = ", ".join(attempts)
        raise ConnectionError(
            f"Could not open webcam index {camera_index} using: {attempted}."
        )

    def camera_read_loop(self, camera_index, camera_backend, stop_event):
        capture = None
        try:
            capture, opened_backend = self.open_camera_capture(
                camera_index, camera_backend
            )
            if stop_event.is_set() or stop_event is not self.camera_stop:
                return

            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            first_frame = self.read_camera_frame_with_warmup(capture)
            if first_frame is None:
                raise ConnectionError(
                    f"Webcam {camera_index} opened but did not return frames."
                )
            self.camera_capture = capture
            self.set_camera_status_safe(
                f"Webcam {camera_index} connected using {opened_backend}", "green"
            )

            frame, summary = self.process_camera_frame(first_frame)
            if frame is not None:
                self.put_latest_camera_frame(frame, summary)

            while not stop_event.is_set():
                success, frame_bgr = capture.read()
                if not success:
                    if stop_event.is_set():
                        break
                    raise ConnectionError(
                        f"Webcam {camera_index} stopped returning frames."
                    )
                frame, summary = self.process_camera_frame(frame_bgr)
                if frame is not None:
                    self.put_latest_camera_frame(frame, summary)

        except Exception as e:
            if not stop_event.is_set():
                self.set_camera_status_safe(self.camera_error_message(e), "red")
        finally:
            if capture is not None:
                capture.release()
            if self.camera_capture is capture:
                self.camera_capture = None

    def process_camera_frame(self, frame_bgr):
        if self.robot_detector is None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb), "OpenCV detection unavailable"

        annotated, detections = self.robot_detector.process(
            frame_bgr,
            mode=self.camera_detection_mode,
            color=self.camera_detection_color,
            minimum_area=self.camera_detection_minimum_area,
        )
        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb), self.robot_detector.summarize(detections)

    @staticmethod
    def camera_error_message(error):
        return (
            f"Webcam connection failed: {error} "
            "Check the selected index, Windows camera permissions, and whether "
            "another application is using the webcam."
        )

    def put_latest_camera_frame(self, frame, summary=""):
        try:
            self.camera_frames.put_nowait((frame, summary))
        except queue.Full:
            try:
                self.camera_frames.get_nowait()
            except queue.Empty:
                pass
            try:
                self.camera_frames.put_nowait((frame, summary))
            except queue.Full:
                pass

    def poll_camera_frames(self):
        if not self.camera_window or not self.camera_window.winfo_exists():
            return

        frame_item = None
        while not self.camera_frames.empty():
            try:
                frame_item = self.camera_frames.get_nowait()
            except queue.Empty:
                break

        if frame_item is not None and self.camera_canvas.winfo_exists():
            frame, summary = frame_item
            width = max(self.camera_canvas.winfo_width(), 1)
            height = max(self.camera_canvas.winfo_height(), 1)
            frame.thumbnail((width, height), Image.Resampling.LANCZOS)
            self.camera_photo = ImageTk.PhotoImage(frame)
            self.camera_canvas.delete("camera_frame")
            self.camera_canvas.create_image(
                width // 2,
                height // 2,
                image=self.camera_photo,
                anchor="center",
                tags="camera_frame",
            )
            if self.camera_detection_label.winfo_exists():
                self.camera_detection_label.config(text=summary)

        self.camera_window.after(50, self.poll_camera_frames)

    def disconnect_camera(self, update_status=True):
        if self.camera_stop is not None:
            self.camera_stop.set()
        self.camera_stop = None

        if self.camera_capture is not None:
            self.camera_capture.release()
        self.camera_capture = None

        if update_status:
            self.set_camera_status("Camera disconnected", "red")

    def close_camera_window(self):
        self.disconnect_camera(update_status=False)
        if self.camera_window and self.camera_window.winfo_exists():
            self.camera_window.destroy()
        self.camera_window = None
        self.camera_canvas = None
        self.camera_status_label = None
        self.camera_detection_label = None
        self.camera_photo = None

    def set_camera_status(self, text, color="black"):
        if self.camera_status_label and self.camera_status_label.winfo_exists():
            self.camera_status_label.config(text=text, fg=color)

    def set_camera_status_safe(self, text, color="black"):
        self.root.after(0, lambda: self.set_camera_status(text, color))

    def ensure_port(self):
        if not self.port or not getattr(self.port, "is_open", False):
            messagebox.showerror("Error", "No active serial port")
            return False
        return True

    def run_thread(self, target):
        threading.Thread(target=target, daemon=True).start()

    def show_error(self, title, text):
        self.root.after(0, lambda: messagebox.showerror(title, text))

    def set_status_safe(self, text, color="black"):
        self.root.after(0, lambda: self.set_status(text, color))
