import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox

from gui_helpers import BaseServoGUI


class ArduinoSerialGUI(BaseServoGUI):
    def __init__(self, root):
        super().__init__(root, title="Arduino Serial Helmholtz GUI", geometry="860x760")
        self.root.minsize(820, 700)

        self.rx_queue = queue.Queue()
        self.stop_reader = threading.Event()
        self.reader_thread = None
        self.streaming = tk.BooleanVar(value=True)
        self.last_sent = ""

        self.axis_vars = {
            "x": tk.IntVar(value=0),
            "y": tk.IntVar(value=0),
            "z": tk.IntVar(value=0),
        }

        self.build_connection(default_port="COM3", default_baud="115200")
        self.build_axis_controls()
        self.build_manual_controls()
        self.build_status()

        self.root.after(50, self.poll_rx_queue)
        self.root.after(120, self.stream_tick)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def connect_port(self):
        self.stop_reader.set()
        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=0.2)

        super().connect_port()

        if self.port and getattr(self.port, "is_open", False):
            self.port.timeout = 0.05
            self.stop_reader.clear()
            self.reader_thread = threading.Thread(target=self.read_loop, daemon=True)
            self.reader_thread.start()
            self.set_status("Connected and reading serial", "green")

    def build_axis_controls(self):
        frame = tk.LabelFrame(self.root, text="PWM Axis Control", padx=10, pady=10)
        frame.pack(fill="x", padx=10, pady=5)

        for row, axis in enumerate(("x", "y", "z")):
            tk.Label(frame, text=axis.upper(), width=4).grid(
                row=row, column=0, sticky="w", pady=8
            )

            scale = tk.Scale(
                frame,
                from_=-255,
                to=255,
                orient="horizontal",
                resolution=1,
                variable=self.axis_vars[axis],
                command=lambda _value: self.schedule_send(),
                length=520,
            )
            scale.grid(row=row, column=1, padx=8, sticky="ew")

            tk.Spinbox(
                frame,
                from_=-255,
                to=255,
                width=8,
                textvariable=self.axis_vars[axis],
                command=self.schedule_send,
            ).grid(row=row, column=2, padx=5)

            tk.Button(
                frame,
                text="Zero",
                width=8,
                command=lambda a=axis: self.set_axis(a, 0),
            ).grid(row=row, column=3, padx=5)

        frame.columnconfigure(1, weight=1)

        button_frame = tk.Frame(frame)
        button_frame.grid(row=3, column=0, columnspan=4, sticky="w", pady=(12, 0))

        tk.Checkbutton(
            button_frame,
            text="Stream values constantly",
            variable=self.streaming,
        ).pack(side="left", padx=(0, 12))

        tk.Button(button_frame, text="Send Now", command=self.send_current).pack(
            side="left", padx=4
        )
        tk.Button(button_frame, text="All Zero", command=self.all_zero).pack(
            side="left", padx=4
        )
        tk.Button(button_frame, text="Ping", command=lambda: self.write_line("PING")).pack(
            side="left", padx=4
        )

    def build_manual_controls(self):
        frame = tk.LabelFrame(self.root, text="Manual Serial Command", padx=10, pady=10)
        frame.pack(fill="x", padx=10, pady=5)

        self.command_entry = tk.Entry(frame)
        self.command_entry.insert(0, "SET 0 0 0")
        self.command_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.command_entry.bind("<Return>", lambda _event: self.send_manual())

        tk.Button(frame, text="Send Line", command=self.send_manual, width=12).pack(
            side="left"
        )

    def set_axis(self, axis, value):
        self.axis_vars[axis].set(value)
        self.schedule_send()

    def all_zero(self):
        for var in self.axis_vars.values():
            var.set(0)
        self.send_current()

    def schedule_send(self):
        if self.streaming.get():
            self.send_current()

    def current_command(self):
        x = self.axis_vars["x"].get()
        y = self.axis_vars["y"].get()
        z = self.axis_vars["z"].get()
        return f"SET {x} {y} {z}"

    def send_current(self):
        self.write_line(self.current_command())

    def send_manual(self):
        command = self.command_entry.get().strip()
        if command:
            self.write_line(command)

    def write_line(self, line):
        self.write_line_to_serial(line, log_tx=True)

    def write_line_to_serial(self, line, log_tx):
        if not self.ensure_port():
            return

        try:
            self.port.write((line.strip() + "\n").encode("ascii"))
            self.last_sent = line.strip()
            if log_tx:
                self.set_status(f"TX: {self.last_sent}", "blue")
        except Exception as e:
            self.set_status("Serial write failed", "red")
            messagebox.showerror("Serial Error", str(e))

    def read_loop(self):
        while not self.stop_reader.is_set():
            try:
                if not self.port or not getattr(self.port, "is_open", False):
                    time.sleep(0.05)
                    continue

                raw = self.port.readline()
                if raw:
                    text = raw.decode("utf-8", errors="replace").strip()
                    if text:
                        self.rx_queue.put(text)
            except Exception as e:
                self.rx_queue.put(f"READ ERROR: {e}")
                time.sleep(0.2)

    def poll_rx_queue(self):
        try:
            while True:
                text = self.rx_queue.get_nowait()
                self.log(f"RX: {text}")
                self.status_label.config(text=f"RX: {text}", fg="green")
        except queue.Empty:
            pass

        self.root.after(50, self.poll_rx_queue)

    def stream_tick(self):
        if self.streaming.get() and self.port and getattr(self.port, "is_open", False):
            self.write_line_to_serial(self.current_command(), log_tx=False)

        self.root.after(120, self.stream_tick)

    def on_close(self):
        self.stop_reader.set()
        try:
            if self.port and getattr(self.port, "is_open", False):
                self.write_line("STOP")
                self.port.close()
        except Exception:
            pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ArduinoSerialGUI(root)
    root.mainloop()
