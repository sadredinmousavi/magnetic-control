"""Closed-loop experiment GUI: connections and moving-ROI calibration."""

import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

from gui_helpers import BaseServoGUI
from ssh_camera import SSHCameraClient, SSHCameraSettings, preferred_camera_mode
from closed_loop_calibration import CalibrationTab
from axis_calibration import axis_target_angles


class ClosedLoopGUI(BaseServoGUI):
    def __init__(self, root):
        super().__init__(
            root,
            title="First Paper Closed-Loop Control",
            geometry="1100x780",
        )
        self.root.minsize(900, 650)
        self.camera_frame_queue = queue.Queue(maxsize=1)
        self.camera_event_queue = queue.Queue()
        self.ssh_camera = SSHCameraClient(
            self.camera_frame_queue, self.camera_event_queue
        )
        self.remote_camera_photo = None
        self.camera_modes = []
        self.mode_settings = None
        self.mode_loading = False
        self.axis_move_lock = threading.Lock()

        self.build_header()
        self.build_workflow()
        self.build_status()

        self.root.after(40, self.poll_camera)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def build_header(self):
        header = tk.Frame(self.root, padx=12, pady=8)
        header.pack(fill="x")
        tk.Label(
            header,
            text="GUI005 — First Paper Closed-Loop Control",
            font=("Segoe UI", 16, "bold"),
        ).pack(anchor="w")
        tk.Label(
            header,
            text="Connect hardware, then calibrate XY and track robots inside a moving ROI",
            fg="#555555",
        ).pack(anchor="w")

    def build_workflow(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=10, pady=(0, 5))

        connection_tab = tk.Frame(notebook)
        notebook.add(connection_tab, text="1. Connections")
        self.build_serial_panel(connection_tab)
        self.build_ssh_camera_panel(connection_tab)

        calibration_tab = tk.Frame(notebook)
        notebook.add(calibration_tab, text="2. Calibration")
        self.calibration = CalibrationTab(calibration_tab, axis_move=self.move_axis_target)
        self.build_future_tabs(notebook)

    def move_axis_target(self, target, case_name, settle, active):
        """Called from the calibration worker; reuse GUI001's serial protocol."""
        from helpers import moveArray
        import time

        with self.axis_move_lock:
            if not active():
                return
            port = self.port
            if port is None or not getattr(port, "is_open", False):
                raise RuntimeError("Connect the serial controller in Connections first.")
            if not self.ssh_camera.is_running:
                raise RuntimeError("Connect the live Pi camera before axis calibration.")
            angles = axis_target_angles(case_name, target)
            if not active():
                return
            if self.port is not port or not getattr(port, "is_open", False):
                raise RuntimeError("The controller disconnected before the move.")
            moveArray(port, angles, 0, 180)
            deadline = time.monotonic() + settle
            while active() and time.monotonic() < deadline:
                time.sleep(min(.05, max(0, deadline - time.monotonic())))

    def build_future_tabs(self, notebook):
        for title, description in (
            ("3. Commands", "Manual commands and sequence loading will be added next."),
            ("4. Detection", "Moving ROI and robot tracking will be added next."),
        ):
            frame = tk.Frame(notebook, padx=24, pady=24)
            notebook.add(frame, text=title, state="disabled")
            tk.Label(frame, text=description).pack(anchor="w")

    def build_serial_panel(self, parent):
        frame = tk.LabelFrame(
            parent, text="Controller — Serial", padx=10, pady=10
        )
        frame.pack(fill="x", padx=10, pady=8)

        tk.Label(frame, text="Port").grid(row=0, column=0, sticky="w")
        self.port_entry = tk.Entry(frame, width=16)
        self.port_entry.insert(0, "COM5")
        self.port_entry.grid(row=0, column=1, padx=(5, 15))

        tk.Label(frame, text="Baudrate").grid(row=0, column=2, sticky="w")
        self.baud_entry = tk.Entry(frame, width=14)
        self.baud_entry.insert(0, "1000000")
        self.baud_entry.grid(row=0, column=3, padx=(5, 15))

        self.serial_connect_button = tk.Button(
            frame, text="Connect", width=12, command=self.connect_port
        )
        self.serial_connect_button.grid(row=0, column=4, padx=4)
        tk.Button(
            frame, text="Disconnect", width=12, command=self.disconnect_port
        ).grid(row=0, column=5, padx=4)

        self.serial_state_label = tk.Label(
            frame, text="Disconnected", fg="red", width=18, anchor="w"
        )
        self.serial_state_label.grid(row=0, column=6, padx=(12, 0), sticky="w")

    def connect_port(self):
        super().connect_port()
        connected = bool(self.port and getattr(self.port, "is_open", False))
        self.serial_state_label.config(
            text="Connected" if connected else "Disconnected",
            fg="green" if connected else "red",
        )

    def disconnect_port(self):
        super().disconnect_port()
        self.serial_state_label.config(text="Disconnected", fg="red")

    def build_ssh_camera_panel(self, parent):
        frame = tk.LabelFrame(
            parent, text="Raspberry Pi Camera — SSH", padx=10, pady=10
        )
        frame.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        frame.columnconfigure(5, weight=1)
        frame.rowconfigure(5, weight=1)

        tk.Label(frame, text="Username").grid(row=0, column=0, sticky="w")
        self.ssh_username_entry = tk.Entry(frame, width=14)
        self.ssh_username_entry.insert(0, "sadra")
        self.ssh_username_entry.grid(row=0, column=1, padx=(5, 12))

        tk.Label(frame, text="IP / hostname").grid(row=0, column=2, sticky="w")
        self.ssh_host_entry = tk.Entry(frame, width=20)
        self.ssh_host_entry.insert(0, "192.168.50.2")
        self.ssh_host_entry.grid(row=0, column=3, padx=(5, 12))

        tk.Label(frame, text="SSH port").grid(row=0, column=4, sticky="w")
        self.ssh_port_entry = tk.Entry(frame, width=8)
        self.ssh_port_entry.insert(0, "22")
        self.ssh_port_entry.grid(row=0, column=5, padx=(5, 12), sticky="w")

        tk.Label(frame, text="Password").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        self.ssh_password_entry = tk.Entry(frame, width=20, show="*")
        self.ssh_password_entry.insert(0, "123456789")
        self.ssh_password_entry.grid(
            row=1, column=1, padx=(5, 12), pady=(8, 0), sticky="w"
        )

        tk.Label(frame, text="Camera mode").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.camera_mode_combo = ttk.Combobox(frame, state="readonly", width=64)
        self.camera_mode_combo.set("Query the Pi to list available sensor modes")
        self.camera_mode_combo.grid(row=2, column=1, columnspan=4, sticky="ew", padx=5, pady=(8, 0))
        self.camera_mode_combo.bind("<<ComboboxSelected>>", self.on_camera_mode_selected)
        fps_frame = tk.Frame(frame)
        fps_frame.grid(row=2, column=5, sticky="w", pady=(8, 0))
        tk.Label(fps_frame, text="FPS").pack(side="left")
        self.camera_fps_entry = tk.Entry(fps_frame, width=7)
        self.camera_fps_entry.insert(0, "30")
        self.camera_fps_entry.pack(side="left", padx=4)
        self.camera_modes_button = tk.Button(frame, text="Load modes", command=self.load_camera_modes)
        self.camera_modes_button.grid(row=2, column=6, columnspan=2, pady=(8, 0))

        self.camera_connect_button = tk.Button(
            frame, text="Connect Camera", width=15, command=self.connect_ssh_camera
        )
        self.camera_connect_button.grid(row=0, column=6, padx=4)
        self.camera_disconnect_button = tk.Button(
            frame,
            text="Disconnect",
            width=12,
            state="disabled",
            command=self.disconnect_ssh_camera,
        )
        self.camera_disconnect_button.grid(row=0, column=7, padx=4)

        self.ssh_state_label = tk.Label(
            frame,
            text="Camera disconnected",
            fg="red",
            anchor="w",
        )
        self.ssh_state_label.grid(
            row=3, column=0, columnspan=8, sticky="ew", pady=(8, 4)
        )
        tk.Label(
            frame,
            text=(
                "Password authentication uses PuTTY plink. Clear the password to "
                "use Windows OpenSSH key/agent authentication instead. Connect "
                "once in a terminal first to trust the Pi host key."
            ),
            fg="#555555",
            anchor="w",
        ).grid(row=4, column=0, columnspan=8, sticky="ew", pady=(0, 4))

        self.remote_camera_canvas = tk.Canvas(
            frame, background="black", highlightthickness=0
        )
        self.remote_camera_canvas.grid(
            row=5, column=0, columnspan=8, sticky="nsew", pady=(4, 0)
        )
        self.remote_camera_canvas.create_text(
            20,
            20,
            text="Connect to the Raspberry Pi to start the camera preview.",
            fill="white",
            anchor="nw",
            tags="placeholder",
        )

    def read_ssh_settings(self, include_mode=True):
        try:
            port = int(self.ssh_port_entry.get().strip())
        except ValueError as error:
            raise ValueError("SSH port must be a whole number.") from error
        mode = None
        framerate = 30.0
        if include_mode:
            index = self.camera_mode_combo.current()
            if not 0 <= index < len(self.camera_modes):
                raise ValueError("Load and select a camera mode first.")
            mode = self.camera_modes[index]
            try:
                framerate = float(self.camera_fps_entry.get().strip())
            except ValueError as error:
                raise ValueError("Camera FPS must be a positive number.") from error
        settings = SSHCameraSettings(
            username=self.ssh_username_entry.get().strip(),
            host=self.ssh_host_entry.get().strip(),
            port=port,
            password=self.ssh_password_entry.get(),
            camera_mode=mode,
            framerate=framerate,
        )
        settings.validate()
        return settings

    def on_camera_mode_selected(self, event=None):
        index = self.camera_mode_combo.current()
        if 0 <= index < len(self.camera_modes):
            fps = min(30.0, self.camera_modes[index].max_fps)
            self.camera_fps_entry.delete(0, "end")
            self.camera_fps_entry.insert(0, f"{fps:g}")

    def load_camera_modes(self, connect_after=False):
        if self.mode_loading:
            return
        if self.ssh_camera.is_running:
            messagebox.showinfo("Camera modes", "Disconnect the camera before querying its modes.")
            return
        try:
            settings = self.read_ssh_settings(include_mode=False)
        except ValueError as error:
            messagebox.showerror("SSH Camera Error", str(error))
            return
        self.mode_loading = True
        self.camera_connect_button.config(state="disabled")
        self.camera_modes_button.config(state="disabled")
        self.camera_mode_combo.config(state="disabled")
        self.ssh_state_label.config(text="Querying camera sensor modes on the Pi...", fg="blue")

        def query():
            try:
                modes = SSHCameraClient.list_modes(settings)
                self.camera_event_queue.put(("modes", settings, modes, connect_after))
            except Exception as error:
                self.camera_event_queue.put(("mode_error", str(error)))

        threading.Thread(target=query, daemon=True).start()

    def connect_ssh_camera(self):
        try:
            base_settings = self.read_ssh_settings(include_mode=False)
            if self.mode_settings != base_settings or not self.camera_modes:
                self.load_camera_modes(connect_after=True)
                return
            settings = self.read_ssh_settings()
            self.ssh_camera.start(settings)
        except Exception as error:
            messagebox.showerror("SSH Camera Error", str(error))
            self.set_status("SSH camera connection failed", "red")
            return
        self.camera_connect_button.config(state="disabled")
        self.camera_disconnect_button.config(state="normal")
        self.camera_modes_button.config(state="disabled")
        self.camera_mode_combo.config(state="disabled")
        self.camera_fps_entry.config(state="disabled")
        target = f"{settings.username}@{settings.host}:{settings.port}"
        self.ssh_state_label.config(
            text=f"Connecting to {target} and waiting for video...", fg="blue"
        )
        self.set_status(
            f"Opening {settings.camera_mode.width} x {settings.camera_mode.height} "
            f"at {settings.framerate:g} fps: {target}", "blue"
        )

    def disconnect_ssh_camera(self):
        self.ssh_camera.stop()
        self.camera_connect_button.config(state="normal")
        self.camera_disconnect_button.config(state="disabled")
        self.restore_camera_mode_controls()
        self.ssh_state_label.config(text="Camera disconnected", fg="red")
        self.set_status("SSH camera disconnected", "red")

    def restore_camera_mode_controls(self):
        self.camera_modes_button.config(state="normal")
        self.camera_mode_combo.config(state="readonly")
        self.camera_fps_entry.config(state="normal")

    def poll_camera(self):
        try:
            while True:
                event = self.camera_event_queue.get_nowait()
                kind = event[0]
                if kind == "modes":
                    self.mode_loading = False
                    self.camera_connect_button.config(state="normal")
                    self.restore_camera_mode_controls()
                    try:
                        current = self.read_ssh_settings(include_mode=False)
                    except ValueError:
                        current = None
                    if current != event[1]:
                        self.ssh_state_label.config(text="SSH settings changed. Load modes again.", fg="red")
                        continue
                    self.mode_settings, self.camera_modes = event[1], event[2]
                    self.camera_mode_combo.config(values=[mode.label for mode in self.camera_modes])
                    preferred = preferred_camera_mode(self.camera_modes)
                    self.camera_mode_combo.current(self.camera_modes.index(preferred))
                    self.on_camera_mode_selected()
                    self.ssh_state_label.config(text=f"Found {len(self.camera_modes)} sensor modes. Choose a mode and connect.", fg="green")
                    if event[3]:
                        self.connect_ssh_camera()
                elif kind == "mode_error":
                    self.mode_loading = False
                    self.camera_connect_button.config(state="normal")
                    self.restore_camera_mode_controls()
                    self.ssh_state_label.config(text="Could not query camera modes", fg="red")
                    messagebox.showerror("Camera modes", event[1])
                elif kind == "connected":
                    self.ssh_state_label.config(
                        text="Camera connected — receiving video", fg="green"
                    )
                    self.set_status("SSH camera is streaming", "green")
                elif kind == "log":
                    self.log(event[1])
                elif kind == "error":
                    self.ssh_state_label.config(text=event[1], fg="red")
                    self.camera_connect_button.config(state="normal")
                    self.camera_disconnect_button.config(state="disabled")
                    self.restore_camera_mode_controls()
                    self.set_status(f"SSH camera error: {event[1]}", "red")
        except queue.Empty:
            pass

        latest_frame = None
        try:
            while True:
                latest_frame = self.camera_frame_queue.get_nowait()
        except queue.Empty:
            pass
        if latest_frame is not None:
            self.show_camera_frame(latest_frame)
            self.calibration.accept_live_frame(latest_frame)
        self.root.after(40, self.poll_camera)

    def show_camera_frame(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        width = max(self.remote_camera_canvas.winfo_width(), 1)
        height = max(self.remote_camera_canvas.winfo_height(), 1)
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        self.remote_camera_photo = ImageTk.PhotoImage(image)
        self.remote_camera_canvas.delete("all")
        self.remote_camera_canvas.create_image(
            width // 2,
            height // 2,
            image=self.remote_camera_photo,
            anchor="center",
        )

    def on_close(self):
        self.calibration.close()
        self.ssh_camera.stop()
        try:
            if self.port and getattr(self.port, "is_open", False):
                self.port.close()
        except Exception:
            pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ClosedLoopGUI(root)
    root.mainloop()

