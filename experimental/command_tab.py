"""GUI005 desired XY overlay and explicit, cancellable sequence execution."""

import json
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
)
from point_control import hardware_case_name, load_hardware_case


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
        bar = tk.Frame(parent)
        bar.pack(fill="x", padx=8, pady=6)
        self.load_button = tk.Button(bar, text="Load XY sequence", command=self.load_sequence)
        self.load_button.pack(side="left", padx=3)
        self.calibration_button = tk.Button(bar, text="Use current calibration", command=self.refresh_calibration)
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
            "Table shows mm; ratio = 1. "
            "Each point is calculated fresh, sent, then held for its wait time. This is timed execution, not arrival feedback. "
            "Stop prevents further commands; it does not undo the last motor pose."
        )).pack(fill="x", padx=10, pady=3)
        table_frame = tk.Frame(parent)
        table_frame.pack(fill="x", padx=8)
        self.table = ttk.Treeview(table_frame, columns=("x", "y", "wait", "zero", "state", "angles"),
                                  show="tree headings", height=6, selectmode="browse")
        self.table.tag_configure("outside", foreground="#b00020")
        self.table.heading("#0", text="Point")
        self.table.column("#0", width=55, stretch=False)
        for key, title, width in (("x", "X (mm)", 75), ("y", "Y (mm)", 75),
                                  ("wait", "Wait (s)", 65), ("zero", "Zero (deg)", 75),
                                  ("state", "State", 95), ("angles", "Calculated motor angles (deg)", 380)):
            self.table.heading(key, text=title)
            self.table.column(key, width=width, stretch=(key == "angles"))
        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.table.pack(side="left", fill="x", expand=True)
        self.table.bind("<<TreeviewSelect>>", self.select_point)
        self.status = tk.Label(parent, text="Load an XY sequence and complete/load Calibration first.", anchor="w", wraplength=1000)
        self.status.pack(fill="x", padx=10, pady=4)
        self.limits_label = tk.Label(parent, text="", anchor="w", justify="left", wraplength=1000)
        self.limits_label.pack(fill="x", padx=10)
        tk.Label(parent, text="Cyan: desired points    Orange: active target    Red: beyond pins    Magenta: calibration rectangle. Select a row or click a target.",
                 anchor="w").pack(fill="x", padx=10)
        self.canvas = tk.Canvas(parent, background="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=(3, 8))
        self.canvas.bind("<Configure>", lambda event: self.render())
        self.canvas.bind("<Button-1>", self.click_target)

    def load_sequence(self):
        if self.running:
            return
        path = filedialog.askopenfilename(title="Load desired XY sequence", initialdir=str(XY_SEQUENCE_DIR),
                                          filetypes=[("XY sequences", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.load_path(path)
        except (ValueError, OSError) as error:
            messagebox.showerror("XY sequence", str(error))
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
            messagebox.showerror("XY sequence units", str(error))

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
                f"{step.wait_seconds:g}", f"{step.zero_angle_deg:g}", "Pending", ""))
        self.refresh_calibration()

    def prepare(self, *, validate=True):
        if not self.steps:
            raise ValueError("Load an XY sequence first.")
        if self.calibration.capture is not None:
            raise ValueError("Select Live camera in Calibration before using it for commands.")
        setup = json.loads(json.dumps(self.calibration.setup_data()))
        if not setup.get("control_case"):
            raise ValueError("Select the hardware case in Calibration first.")
        pixels = project_sequence(self.steps, setup, validate=validate)
        if self.frame is not None and list(self.frame.shape[1::-1]) != setup["video_size_px"]:
            raise ValueError("Live camera size differs from calibration. Recalibrate or load a matching setup.")
        return setup, pixels

    def refresh_calibration(self):
        if self.running:
            return
        try:
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
            self.status.config(text=f"Not ready: {error}")
            self.limits_label.config(text="")
        self.render()

    def accept_live_frame(self, frame):
        self.frame = frame.copy()
        self.frame_received_at = time.monotonic()
        if self.setup is not None and list(frame.shape[1::-1]) != self.setup["video_size_px"]:
            self.stop_event.set()
            self.pixels = None
            self.status.config(text="Camera resolution changed. Sequence stopped; load matching calibration before running again.")
        if self.visible():
            self.render()

    def select_point(self, event=None):
        if not self.running and self.table.selection():
            self.active_index = int(self.table.selection()[0])
            self.render()

    def show_points(self):
        self.show_all = True
        if self.running:
            self.render()
        else:
            self.refresh_calibration()

    def click_target(self, event):
        if self.running or self.pixels is None or not hasattr(self, "screen_points"):
            return
        distances = np.linalg.norm(self.screen_points - [event.x, event.y], axis=1)
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
        except (ValueError, TypeError, KeyError, ImportError) as error:
            messagebox.showerror("Cannot run sequence", str(error))
            return
        outside, _ = point_limits(pixels, setup)
        extrapolated = sum(bool(outside[index]) for index in indices)
        description = f"selected point {indices[0] + 1}" if selected_only else f"all {len(indices)} points"
        warning = f"\nWARNING: {extrapolated} target(s) beyond pins. Mapping is extrapolated; confirm they are physically safe." if extrapolated else ""
        if not messagebox.askyesno("Run XY targets", f"Run {description} using {setup['control_case']}?\nInput units: {self.input_units}; table: mm; calibration: cm.\nRatio 1; wait and zero from each XY row.\nConfirm the units and physical hardware match.{warning}"):
            return
        self.setup, self.pixels = setup, pixels
        self.running = True
        self.active_index = None
        self.stop_event.clear()
        self.job += 1
        job = self.job
        steps = tuple(self.steps[index] for index in indices)
        self.run_indices = indices
        self.set_running(True)
        for button in (self.load_button, self.calibration_button, self.run_button, self.run_selected_button):
            button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.units_combo.config(state="disabled")
        self.extrapolation_check.config(state="disabled")
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
                                              self.events.put((job, kind, indices[index], result)))
                self.events.put((job, outcome, None, None))
            except Exception as error:
                self.events.put((job, "error", None, str(error)))

        threading.Thread(target=worker, daemon=True).start()
        self.after_id = self.parent.after(50, self.poll)

    def stop(self):
        if self.running:
            self.stop_event.set()
            self.stop_button.config(state="disabled")
            self.status.config(text="Stop requested. Unsent commands will be cancelled; an ongoing calculation may take time to return. Last pose stays applied.")

    def poll(self):
        self.after_id = None
        try:
            while True:
                job, kind, index, result = self.events.get_nowait()
                if job != self.job:
                    continue
                if kind in ("calculating", "sent", "completed"):
                    self.active_index = index
                    state = {"calculating": "Calculating", "sent": "Holding", "completed": "Completed"}[kind]
                    self.table.set(str(index), "state", state)
                    self.table.see(str(index))
                    if result is not None:
                        self.table.set(str(index), "angles", ", ".join(f"{angle:.2f}" for angle in result["angles_deg"]))
                    target = self.steps[index].target_xy_m
                    self.status.config(text=f"Point {index + 1}/{len(self.steps)}: {state.lower()}; desired XY ({target[0] * 1000:g}, {target[1] * 1000:g}) mm.")
                    self.render()
                else:
                    self.running = False
                    self.set_running(False)
                    for button in (self.load_button, self.calibration_button, self.run_button, self.run_selected_button):
                        button.config(state="normal")
                    self.stop_button.config(state="disabled")
                    self.units_combo.config(state="readonly")
                    self.extrapolation_check.config(state="normal")
                    if kind != "finished" and self.active_index is not None:
                        self.table.set(str(self.active_index), "state", "Failed" if kind == "error" else "Stopped")
                    self.status.config(text=("Sequence complete." if kind == "finished" else
                                             f"Sequence failed: {result}. No further points sent." if kind == "error" else
                                             "Sequence stopped (Stop, camera loss, or controller disconnect). Last sent pose remains applied."))
                    if kind == "error":
                        messagebox.showerror("Sequence stopped", str(result))
        except queue.Empty:
            pass
        if self.running:
            self.after_id = self.parent.after(50, self.poll)

    def render(self):
        self.canvas.delete("all")
        if self.frame is None:
            return
        h, w = self.frame.shape[:2]
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        low, high = np.array([0., 0.]), np.array([float(w), float(h)])
        if self.pixels is not None and self.show_all:
            low = np.minimum(low, self.pixels.min(axis=0) - [15, 15])
            high = np.maximum(high, self.pixels.max(axis=0) + [15, 15])
        extent = high - low
        scale = min(cw / extent[0], ch / extent[1])
        dw, dh = max(1, round(w * scale)), max(1, round(h * scale))
        ox, oy = (np.array([cw, ch]) - extent * scale) / 2 - low * scale
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)).resize((dw, dh)))
        self.canvas.create_image(ox + dw / 2, oy + dh / 2, image=self.photo, anchor="center")
        if self.pixels is None or self.setup is None:
            return
        corners = np.asarray(self.setup["corners_px"]) * scale + [ox, oy]
        self.canvas.create_polygon(*corners.ravel(), fill="", outline="#ff00ff", width=1)
        points = self.pixels * scale + [ox, oy]
        self.screen_points = points
        if len(points) > 1:
            self.canvas.create_line(*points.ravel(), fill="#00ffff", width=1, dash=(4, 3))
        outside, _ = point_limits(self.pixels, self.setup)
        for index, (x, y) in enumerate(points):
            self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#ff4444" if outside[index] else "#00ffff", outline="black", width=1)
        if self.active_index is not None and self.active_index < len(points):
            x, y = points[self.active_index]
            self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, outline="#ff9900", width=1)
            self.canvas.create_line(x - 8, y, x + 8, y, fill="#ff9900", width=1)
            self.canvas.create_line(x, y - 8, x, y + 8, fill="#ff9900", width=1)
            self.canvas.create_text(x + 10, y - 10, text=f"P{self.active_index + 1}", fill="#ff9900", anchor="sw")

    def close(self):
        self.stop_event.set()
        self.job += 1
        if self.after_id is not None:
            self.parent.after_cancel(self.after_id)
            self.after_id = None
