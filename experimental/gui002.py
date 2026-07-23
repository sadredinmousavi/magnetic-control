import ast
import tkinter as tk
from tkinter import messagebox

from gui_helpers import BaseServoGUI
from helpers import move, moveArray, setID, setLedStatus, factoryReset


class ServoTestGUI(BaseServoGUI):
    def __init__(self, root):
        super().__init__(root, title="Servo Test GUI", geometry="760x760")
        self.root.minsize(760, 760)
        self.root.resizable(True, True)

        self.shared_servo_id = tk.IntVar(value=1)

        self.build_connection()
        self.build_shared_servo_id()
        self.build_custom_move()
        self.build_led()
        self.build_single_move()
        self.build_change_id()
        self.build_factory_reset()
        self.build_status()

    def build_shared_servo_id(self):
        frame = tk.LabelFrame(self.root, text="Shared Servo ID", padx=10, pady=10)
        frame.pack(fill="x", padx=10, pady=5)

        tk.Label(frame, text="Servo ID (1-8)").grid(row=0, column=0, sticky="w")

        self.shared_id_spinbox = tk.Spinbox(
            frame,
            from_=1,
            to=8,
            width=10,
            textvariable=self.shared_servo_id
        )
        self.shared_id_spinbox.grid(row=0, column=1, padx=5, sticky="w")

        tk.Label(
            frame,
            text="Used by LED Control, Move One Motor, and Factory Reset",
            fg="gray"
        ).grid(row=0, column=2, padx=10, sticky="w")

    def get_shared_servo_id(self):
        try:
            servo_id = int(self.shared_id_spinbox.get().strip())
            if not 1 <= servo_id <= 8:
                raise ValueError("Servo ID must be between 1 and 8.")
            return servo_id
        except Exception:
            raise ValueError("Please enter a valid Servo ID between 1 and 8.")

    def build_custom_move(self):
        frame = tk.LabelFrame(self.root, text="Custom moveArray", padx=10, pady=10)
        frame.pack(fill="x", padx=10, pady=5)

        tk.Label(frame, text="Angles").grid(row=0, column=0, sticky="w")
        self.angles_entry = tk.Entry(frame, width=85)
        self.angles_entry.insert(
            0, "[-85.28, 68.47, -85.25, 5.16, -8.61, -4.22, -1.85, 6.38]"
        )
        self.angles_entry.grid(row=0, column=1, columnspan=4, padx=5, pady=5)

        tk.Label(frame, text="Wait").grid(row=1, column=0, sticky="e")
        self.wait_entry = tk.Entry(frame, width=10)
        self.wait_entry.insert(0, "2")
        self.wait_entry.grid(row=1, column=1, sticky="w")

        tk.Label(frame, text="Zero Angle").grid(row=1, column=2, sticky="e")
        self.zero_entry = tk.Entry(frame, width=10)
        self.zero_entry.insert(0, "180")
        self.zero_entry.grid(row=1, column=3, sticky="w")

        tk.Button(frame, text="Run Custom", command=self.run_custom).grid(
            row=1, column=4, padx=5
        )

    def build_led(self):
        frame = tk.LabelFrame(self.root, text="LED Control", padx=10, pady=10)
        frame.pack(fill="x", padx=10, pady=5)

        tk.Label(frame, text="Servo ID").grid(row=0, column=0)
        tk.Label(frame, textvariable=self.shared_servo_id, width=10, relief="sunken").grid(
            row=0, column=1, padx=5
        )

        tk.Button(frame, text="LED ON", command=lambda: self.set_led(1)).grid(
            row=0, column=2, padx=5
        )
        tk.Button(frame, text="LED OFF", command=lambda: self.set_led(0)).grid(
            row=0, column=3, padx=5
        )

    def build_single_move(self):
        frame = tk.LabelFrame(self.root, text="Move One Motor", padx=10, pady=10)
        frame.pack(fill="x", padx=10, pady=5)

        tk.Label(frame, text="Servo ID").grid(row=0, column=0)
        tk.Label(frame, textvariable=self.shared_servo_id, width=10, relief="sunken").grid(
            row=0, column=1, padx=5
        )

        tk.Label(frame, text="Angle").grid(row=0, column=2)
        self.move_angle_entry = tk.Entry(frame, width=10)
        self.move_angle_entry.insert(0, "0")
        self.move_angle_entry.grid(row=0, column=3, padx=5)

        tk.Label(frame, text="Zero Angle").grid(row=0, column=4)
        self.move_zero_entry = tk.Entry(frame, width=10)
        self.move_zero_entry.insert(0, "180")
        self.move_zero_entry.grid(row=0, column=5, padx=5)

        tk.Button(frame, text="Move", command=self.move_one_motor).grid(
            row=0, column=6, padx=8
        )

        preset_frame = tk.Frame(frame)
        preset_frame.grid(row=1, column=0, columnspan=7, pady=(8, 0), sticky="w")

        preset_angles = [0, 45, 90, 135, 180, -45, -90, -135]

        for i, ang in enumerate(preset_angles):
            tk.Button(
                preset_frame,
                text=str(ang),
                width=6,
                command=lambda a=ang: self.set_preset_angle(a)
            ).grid(row=0, column=i, padx=3, pady=2)

    def build_change_id(self):
        frame = tk.LabelFrame(self.root, text="Change ID", padx=10, pady=10)
        frame.pack(fill="x", padx=10, pady=5)

        tk.Label(frame, text="Old ID").grid(row=0, column=0)
        self.old_id_entry = tk.Entry(frame, width=10)
        self.old_id_entry.insert(0, "2")
        self.old_id_entry.grid(row=0, column=1, padx=5)

        tk.Label(frame, text="New ID").grid(row=0, column=2)
        self.new_id_entry = tk.Entry(frame, width=10)
        self.new_id_entry.insert(0, "6")
        self.new_id_entry.grid(row=0, column=3, padx=5)

        tk.Button(frame, text="Set ID", command=self.change_id).grid(
            row=0, column=4, padx=5
        )

    def build_factory_reset(self):
        frame = tk.LabelFrame(self.root, text="Factory Reset", padx=10, pady=10)
        frame.pack(fill="x", padx=10, pady=5)

        tk.Label(frame, text="Servo ID").grid(row=0, column=0)
        tk.Label(frame, textvariable=self.shared_servo_id, width=10, relief="sunken").grid(
            row=0, column=1, padx=5
        )

        tk.Button(frame, text="Factory Reset", command=self.reset_servo).grid(
            row=0, column=2, padx=8
        )

        tk.Label(
            frame,
            text="Warning: reset may restore the servo ID to its default value.",
            fg="red",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 0))

    def run_custom(self):
        if not self.ensure_port():
            return

        def worker():
            try:
                angles = ast.literal_eval(self.angles_entry.get().strip())
                wait = float(self.wait_entry.get().strip())
                zero_ang = float(self.zero_entry.get().strip())

                self.set_status_safe("Running custom move...", "blue")
                moveArray(self.port, angles, wait, zero_ang)
                self.set_status_safe("Custom move done", "green")

            except Exception as e:
                self.set_status_safe("Custom move failed", "red")
                self.show_error("Error", str(e))

        self.run_thread(worker)

    def set_led(self, state):
        if not self.ensure_port():
            return

        try:
            servo_id = self.get_shared_servo_id()
            setLedStatus(self.port, servo_id, state)
            self.set_status(
                f"LED {'ON' if state else 'OFF'} for servo {servo_id}", "blue"
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def set_preset_angle(self, angle):
        self.move_angle_entry.delete(0, tk.END)
        self.move_angle_entry.insert(0, str(angle))

    def move_one_motor(self):
        if not self.ensure_port():
            return

        try:
            servo_id = self.get_shared_servo_id()
            angle = float(self.move_angle_entry.get().strip())
            zero_ang = float(self.move_zero_entry.get().strip())

            move(self.port, servo_id, angle + zero_ang)

            self.set_status(
                f"Moved servo {servo_id} to {angle} degrees", "blue"
            )

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def change_id(self):
        if not self.ensure_port():
            return

        try:
            old_id = int(self.old_id_entry.get().strip())
            new_id = int(self.new_id_entry.get().strip())
            setID(self.port, old_id, new_id)
            self.set_status(f"Changed ID from {old_id} to {new_id}", "blue")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def reset_servo(self):
        if not self.ensure_port():
            return

        try:
            servo_id = self.get_shared_servo_id()
            factoryReset(self.port, servo_id)
            self.set_status(f"Factory reset sent to servo {servo_id}", "blue")
        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = ServoTestGUI(root)
    root.mainloop()
