import threading
import tkinter as tk
from tkinter import messagebox

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

    # Linux default: /dev/ttyS0
    def build_connection(self, default_port="COM3", default_baud="1000000"):
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
