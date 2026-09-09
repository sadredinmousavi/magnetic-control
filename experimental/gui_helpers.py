import io
import queue
import threading
import tkinter as tk
from tkinter import messagebox
from urllib.parse import urlparse
from urllib.request import (
    HTTPBasicAuthHandler,
    HTTPDigestAuthHandler,
    HTTPPasswordMgrWithDefaultRealm,
    ProxyHandler,
    Request,
    build_opener,
)

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None

try:
    import serial
except ImportError:
    serial = None


class BaseServoGUI:
    def __init__(self, root, title="Servo GUI", geometry="800x600"):
        self.root = root
        self.root.title(title)
        self.root.geometry(geometry)

        self.port = None
        self.camera_window = None
        self.camera_canvas = None
        self.camera_status_label = None
        self.camera_response = None
        self.camera_stop = None
        self.camera_frames = queue.Queue(maxsize=1)
        self.camera_photo = None

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
            tk.Button(frame, text="IP Camera", command=self.open_camera_window).grid(
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
        self.camera_window.title("IP Webcam Viewer")
        self.camera_window.geometry("960x720")
        self.camera_window.minsize(640, 480)
        self.camera_window.protocol("WM_DELETE_WINDOW", self.close_camera_window)

        controls = tk.Frame(self.camera_window, padx=10, pady=10)
        controls.pack(fill="x")
        controls.columnconfigure(1, weight=1)

        tk.Label(controls, text="Camera URL").grid(row=0, column=0, sticky="w")
        self.camera_url_entry = tk.Entry(controls)
        self.camera_url_entry.insert(0, "http://10.233.73.201:8080/video")
        self.camera_url_entry.grid(
            row=0, column=1, columnspan=4, sticky="ew", padx=8, pady=4
        )

        tk.Label(controls, text="Username").grid(row=1, column=0, sticky="w")
        self.camera_username_entry = tk.Entry(controls, width=24)
        self.camera_username_entry.grid(row=1, column=1, sticky="w", padx=8, pady=4)

        tk.Label(controls, text="Password").grid(row=1, column=2, sticky="e")
        self.camera_password_entry = tk.Entry(controls, width=24, show="*")
        self.camera_password_entry.grid(row=1, column=3, sticky="w", padx=8, pady=4)

        tk.Button(controls, text="Connect", command=self.connect_camera).grid(
            row=1, column=4, padx=4
        )
        tk.Button(controls, text="Disconnect", command=self.disconnect_camera).grid(
            row=1, column=5, padx=4
        )

        self.camera_status_label = tk.Label(
            self.camera_window,
            text="Enter the phone's IP Webcam video URL, then press Connect.",
            fg="blue",
            anchor="w",
            padx=10,
        )
        self.camera_status_label.pack(fill="x")

        self.camera_canvas = tk.Canvas(
            self.camera_window, background="black", highlightthickness=0
        )
        self.camera_canvas.pack(fill="both", expand=True, padx=10, pady=10)
        self.camera_window.after(50, self.poll_camera_frames)

    def connect_camera(self):
        url = self.camera_url_entry.get().strip()
        username = self.camera_username_entry.get().strip()
        password = self.camera_password_entry.get()
        parsed_url = urlparse(url)
        if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
            messagebox.showerror(
                "Camera Error",
                "Enter a valid URL, for example:\nhttp://192.168.1.100:8080/video",
            )
            return

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
            args=(url, username, password, stop_event),
            daemon=True,
        ).start()

    def camera_read_loop(self, url, username, password, stop_event):
        response = None
        try:
            request = Request(url, headers={"User-Agent": "Magnetic-Control-GUI"})
            # LAN camera streams should connect directly. Windows proxy settings can
            # otherwise route private IP addresses through a blocked proxy socket.
            handlers = [ProxyHandler({})]
            if username:
                password_manager = HTTPPasswordMgrWithDefaultRealm()
                password_manager.add_password(None, url, username, password)
                handlers.extend(
                    [
                        HTTPDigestAuthHandler(password_manager),
                        HTTPBasicAuthHandler(password_manager),
                    ]
                )
            direct_opener = build_opener(*handlers)
            response = direct_opener.open(request, timeout=10)
            if stop_event.is_set() or stop_event is not self.camera_stop:
                return

            self.camera_response = response
            self.set_camera_status_safe("Camera connected", "green")
            data = b""

            while not stop_event.is_set():
                chunk = response.read(4096)
                if not chunk:
                    raise ConnectionError("The camera stream ended.")

                data += chunk
                start = data.find(b"\xff\xd8")
                end = data.find(b"\xff\xd9", start + 2) if start >= 0 else -1
                if start >= 0 and end >= 0:
                    jpeg = data[start : end + 2]
                    data = data[end + 2 :]
                    frame = Image.open(io.BytesIO(jpeg))
                    frame.load()
                    frame = frame.convert("RGB")
                    self.put_latest_camera_frame(frame)
                elif len(data) > 4_000_000:
                    data = data[-1_000_000:]

        except Exception as e:
            if not stop_event.is_set():
                self.set_camera_status_safe(self.camera_error_message(e), "red")
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass
            if self.camera_response is response:
                self.camera_response = None

    @staticmethod
    def camera_error_message(error):
        reason = getattr(error, "reason", error)
        winerror = getattr(reason, "winerror", None)
        if winerror == 10013 or "WinError 10013" in str(error):
            return (
                "The local camera connection was blocked. If a VPN is active, enable "
                "LAN traffic (Windscribe: Preferences > Connection > Allow LAN "
                "Traffic), reconnect the VPN, and retry. Also check firewall/security "
                "software if needed."
            )
        return f"Camera connection failed: {error}"

    def put_latest_camera_frame(self, frame):
        try:
            self.camera_frames.put_nowait(frame)
        except queue.Full:
            try:
                self.camera_frames.get_nowait()
            except queue.Empty:
                pass
            try:
                self.camera_frames.put_nowait(frame)
            except queue.Full:
                pass

    def poll_camera_frames(self):
        if not self.camera_window or not self.camera_window.winfo_exists():
            return

        frame = None
        while not self.camera_frames.empty():
            try:
                frame = self.camera_frames.get_nowait()
            except queue.Empty:
                break

        if frame is not None and self.camera_canvas.winfo_exists():
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

        self.camera_window.after(50, self.poll_camera_frames)

    def disconnect_camera(self, update_status=True):
        if self.camera_stop is not None:
            self.camera_stop.set()
        self.camera_stop = None

        if self.camera_response is not None:
            try:
                self.camera_response.close()
            except Exception:
                pass
        self.camera_response = None

        if update_status:
            self.set_camera_status("Camera disconnected", "red")

    def close_camera_window(self):
        self.disconnect_camera(update_status=False)
        if self.camera_window and self.camera_window.winfo_exists():
            self.camera_window.destroy()
        self.camera_window = None
        self.camera_canvas = None
        self.camera_status_label = None
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
