"""Closed-loop experiment GUI: connections and moving-ROI calibration."""

import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

from gui_helpers import BaseServoGUI
from ssh_camera import SSHCameraClient, SSHCameraSettings, preferred_camera_mode
from closed_loop_calibration import CalibrationTab
from point_control import PointCommander
from command_tab import CommandTab
from camera_popout import show_view_dialog
from local_camera import LocalCameraClient


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
        self.local_frame_queue, self.local_event_queue = queue.Queue(maxsize=1), queue.Queue()
        self.local_camera = LocalCameraClient(self.local_frame_queue, self.local_event_queue)
        self.webcam_requested_size = None
        self.camera_source_var = tk.StringVar(value='Pi SSH')
        self.selected_camera_source = 'Pi SSH'
        self.camera_source_key = 'Pi SSH'
        self.ssh_camera = SSHCameraClient(
            self.camera_frame_queue, self.camera_event_queue
        )
        self.remote_camera_photo = None
        self.camera_modes = []
        self.mode_settings = None
        self.mode_loading = False
        self.point_commander = PointCommander(lambda: self.port)

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
        self.connection_visible = lambda: notebook.select() == str(connection_tab)
        self.build_serial_panel(connection_tab)
        self.build_camera_source_panel(connection_tab)
        self.build_ssh_camera_panel(connection_tab)

        calibration_tab = tk.Frame(notebook)
        notebook.add(calibration_tab, text="2. Calibration")
        self.calibration = CalibrationTab(calibration_tab, axis_move=self.move_axis_target,
                                          axis_prepare=self.point_commander.prepare_axes,
                                          visible=lambda: notebook.select() == str(calibration_tab))
        command_tab = tk.Frame(notebook)
        notebook.add(command_tab, text="3. Commands")
        self.commands = CommandTab(
            command_tab, self.calibration, self.point_commander, self.selected_camera_running,
            set_running=lambda running: (self.calibration.set_command_running(running),
                                         notebook.tab(calibration_tab, state="disabled" if running else "normal")),
            visible=lambda: notebook.select() == str(command_tab),
        )
        notebook.bind("<<NotebookTabChanged>>", lambda event: self.commands.refresh_calibration()
                      if notebook.select() == str(command_tab) else None)
        self.build_future_tabs(notebook)

    def move_axis_target(self, target, case_name, active):
        """Calibration policy; reusable synthesis/motion lives in PointCommander."""
        if not active():
            return
        result = self.point_commander.move_prepared_axis(
            case_name, target,
            settle_seconds=0, active=active,
        )
        if result is not None:
            angles = ", ".join(f"{angle:.2f}" for angle in result["angles_deg"])
            self.camera_event_queue.put(("log", f"Point {target} m, ratio 1: prepared magnet angles [{angles}] deg"))
        return result

    def build_future_tabs(self, notebook):
        for title, description in (
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

    def selected_camera_running(self):
        return self.local_camera.is_running if self.selected_camera_source == 'Iriun / Local webcam' else self.ssh_camera.is_running

    def build_camera_source_panel(self, parent):
        bar = tk.LabelFrame(parent, text='Camera source — shared by Calibration and Commands', padx=8, pady=4)
        bar.pack(fill='x', padx=10, pady=4)
        controls = tk.Frame(bar)
        controls.pack(fill='x')
        source = ttk.Combobox(controls, textvariable=self.camera_source_var, values=('Pi SSH', 'Iriun / Local webcam'),
                              state='readonly', width=24)
        source.pack(side='left', padx=4)
        source.bind('<<ComboboxSelected>>', self.change_camera_source)
        tk.Label(controls, text='Webcam index').pack(side='left', padx=4)
        self.webcam_index_var = tk.StringVar(value='0')
        tk.Spinbox(controls, from_=0, to=10, textvariable=self.webcam_index_var, width=3).pack(side='left')
        tk.Button(controls, text='Connect webcam', command=self.connect_local_camera).pack(side='left', padx=4)
        tk.Button(controls, text='Disconnect webcam', command=self.disconnect_local_camera).pack(side='left', padx=4)
        self.local_state_label = tk.Label(controls, text='Start Iriun on the PC and phone first; try index 0, 1, 2…', anchor='w', wraplength=380)
        self.local_state_label.pack(side='left', padx=6)
        resolution = tk.Frame(bar)
        resolution.pack(fill='x', pady=(4,0))
        tk.Label(resolution, text='Requested resolution').pack(side='left', padx=4)
        self.webcam_resolution_var = tk.StringVar(value='Auto')
        self.webcam_resolution_combo = ttk.Combobox(resolution, textvariable=self.webcam_resolution_var, state='readonly', width=14,
            values=('Auto','640x480','1280x720','1920x1080','2560x1440','3840x2160','Custom'))
        self.webcam_resolution_combo.pack(side='left', padx=4)
        self.webcam_resolution_combo.bind('<<ComboboxSelected>>', self.on_webcam_resolution_selected)
        self.webcam_width_var, self.webcam_height_var = tk.StringVar(value='1920'), tk.StringVar(value='1080')
        tk.Label(resolution, text='Custom width').pack(side='left', padx=4)
        self.webcam_width_entry = tk.Entry(resolution, textvariable=self.webcam_width_var, width=6, state='disabled')
        self.webcam_width_entry.pack(side='left')
        tk.Label(resolution, text='Height').pack(side='left', padx=4)
        self.webcam_height_entry = tk.Entry(resolution, textvariable=self.webcam_height_var, width=6, state='disabled')
        self.webcam_height_entry.pack(side='left')
        tk.Label(resolution, text='Requests driver mode, not image resizing. Disconnect/reconnect to apply.', foreground='#555555').pack(side='left', padx=8)

    def on_webcam_resolution_selected(self, event=None):
        state = 'normal' if self.webcam_resolution_var.get() == 'Custom' else 'disabled'
        self.webcam_width_entry.config(state=state)
        self.webcam_height_entry.config(state=state)

    def read_webcam_resolution(self):
        choice = self.webcam_resolution_var.get()
        if choice == 'Auto':
            return None
        if choice == 'Custom':
            width, height = int(self.webcam_width_var.get()), int(self.webcam_height_var.get())
        else:
            width, height = map(int, choice.split('x'))
        if not 1 <= width <= 16384 or not 1 <= height <= 16384:
            raise ValueError('Resolution must contain whole numbers between 1 and 16384.')
        return width, height

    @staticmethod
    def drain_frames(frames):
        try:
            while True:
                frames.get_nowait()
        except queue.Empty:
            pass

    def reset_camera_geometry(self, source_key):
        self.commands.stop_event.set()
        self.commands.preparation_stop.set()
        self.commands.stop_recording()
        self.calibration.frame = self.calibration.frame_received_at = None
        self.calibration.corners, self.calibration.circle = [], None
        self.calibration.reset_axes()
        self.calibration.clear_tracking()
        self.calibration.camera_source_key = source_key
        self.calibration.setup_revision += 1
        self.commands.frame = self.commands.frame_received_at = None
        self.commands.setup = self.commands.pixels = self.commands.tracking_tracker = None
        self.commands.command_roi = None
        self.commands.roi_override = False
        self.commands.tracking_detections, self.commands.tracking_group_xy = [], None
        self.commands.tracking_signature = None
        self.commands.render_revision += 1
        for tab in (self.calibration, self.commands):
            tab.canvas.delete('all')
            popup = tab.camera_popout
            if popup is not None and popup.is_open:
                popup.canvas.delete('all')
        self.remote_camera_canvas.delete('all')
        self.camera_source_key = source_key
        self.calibration.status.config(text='Camera source changed. Recalibrate or load a setup matching this camera.')

    def change_camera_source(self, event=None):
        source = self.camera_source_var.get()
        if source == self.selected_camera_source:
            return
        self.ssh_camera.stop()
        self.local_camera.stop()
        self.drain_frames(self.camera_frame_queue)
        self.drain_frames(self.local_frame_queue)
        self.drain_frames(self.local_event_queue)
        self.selected_camera_source = source
        self.reset_camera_geometry(source)
        self.camera_connect_button.config(state='normal' if source == 'Pi SSH' else 'disabled')
        self.restore_camera_mode_controls()
        self.camera_disconnect_button.config(state='disabled')
        self.ssh_state_label.config(text='Pi disconnected; camera source changed', fg='red')
        self.set_status(f'Selected {source}. Connect this source; both phases use it.', 'blue')

    def connect_local_camera(self):
        if self.selected_camera_source != 'Iriun / Local webcam':
            show_view_dialog(self, messagebox.showerror, 'Camera source', 'Select Iriun / Local webcam first.')
            return
        try:
            index = int(self.webcam_index_var.get())
            requested = self.read_webcam_resolution()
            if self.local_camera.thread is not None and self.local_camera.thread.is_alive():
                raise RuntimeError('Disconnect webcam first, then reconnect to apply resolution.')
            key = f'Iriun / Local webcam:{index}'
            if key != self.camera_source_key or requested != self.webcam_requested_size:
                self.reset_camera_geometry(key)
            self.drain_frames(self.local_frame_queue)
            self.drain_frames(self.local_event_queue)
            self.local_camera.start(index, **({} if requested is None else dict(width=requested[0], height=requested[1])))
            self.webcam_requested_size = requested
            self.local_state_label.config(text=f'Opening webcam {index}…', fg='blue')
        except (ValueError, RuntimeError) as error:
            show_view_dialog(self, messagebox.showerror, 'Webcam', str(error))

    def disconnect_local_camera(self):
        self.local_camera.stop()
        self.drain_frames(self.local_frame_queue)
        self.local_state_label.config(text='Webcam disconnected', fg='red')

    def build_ssh_camera_panel(self, parent):
        frame = tk.LabelFrame(
            parent, text="Raspberry Pi Camera — SSH", padx=10, pady=10
        )
        frame.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        self.pi_camera_panel = frame
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
            show_view_dialog(self, messagebox.showerror, "SSH Camera Error", str(error))
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
        if getattr(self, 'selected_camera_source', 'Pi SSH') != 'Pi SSH':
            return
        try:
            base_settings = self.read_ssh_settings(include_mode=False)
            if self.mode_settings != base_settings or not self.camera_modes:
                self.load_camera_modes(connect_after=True)
                return
            settings = self.read_ssh_settings()
            self.ssh_camera.start(settings)
        except Exception as error:
            show_view_dialog(self, messagebox.showerror, "SSH Camera Error", str(error))
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
        selected = getattr(self, 'selected_camera_source', 'Pi SSH') == 'Pi SSH'
        self.camera_modes_button.config(state="normal" if selected else 'disabled')
        self.camera_mode_combo.config(state="readonly" if selected else 'disabled')
        self.camera_fps_entry.config(state="normal" if selected else 'disabled')

    def poll_camera(self):
        started = time.monotonic()
        if hasattr(self, 'local_event_queue'):
            try:
                while True:
                    kind, message = self.local_event_queue.get_nowait()
                    if self.selected_camera_source == 'Iriun / Local webcam':
                        color = 'darkorange' if kind == 'connected' and 'different size' in message else 'green' if kind == 'connected' else 'red'
                        self.local_state_label.config(text=message, fg=color)
            except queue.Empty:
                pass
        try:
            while True:
                event = self.camera_event_queue.get_nowait()
                kind = event[0]
                if getattr(self, 'selected_camera_source', 'Pi SSH') != 'Pi SSH' and kind != 'log':
                    if kind in ('modes', 'mode_error'):
                        self.mode_loading = False
                    continue
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
                    show_view_dialog(self, messagebox.showerror, "Camera modes", event[1])
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
                frames = self.local_frame_queue if getattr(self, 'selected_camera_source', 'Pi SSH') == 'Iriun / Local webcam' else self.camera_frame_queue
                latest_frame = frames.get_nowait()
        except queue.Empty:
            pass
        if latest_frame is not None:
            popout_open = any(getattr(tab, 'camera_popout', None) is not None and tab.camera_popout.is_open
                              for tab in (self.calibration, self.commands))
            if getattr(self, 'connection_visible', lambda: True)() and not popout_open:
                self.show_camera_frame(latest_frame)
            self.calibration.accept_live_frame(latest_frame)
            self.commands.accept_live_frame(latest_frame)
        self.commands.check_camera_health()
        try:
            fps = max(1., min(15., float(self.calibration.live_fps_var.get())))
            if not 1 <= fps <= 15:
                fps = 5.
        except (ValueError, TypeError):
            fps = 5.
        delay = max(40, round(1000 / fps - (time.monotonic() - started) * 1000))
        self.root.after(delay, self.poll_camera)

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
        self.commands.close()
        self.calibration.close()
        self.ssh_camera.stop()
        self.local_camera.stop()
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

