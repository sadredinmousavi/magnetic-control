import ast
import os
import tkinter as tk
from tkinter import filedialog, messagebox

from gui_helpers import BaseServoGUI
from helpers import moveArray


class PoseGUI(BaseServoGUI):
    def __init__(self, root):
        super().__init__(root, title="Servo Defined Poses", geometry="760x680")

        self.pose_rows = []
        self.abort_requested = False
        self.is_running_all = False

        self.build_connection()
        self.build_pose_area()
        self.build_status()

        self.load_default_rows()

    def build_pose_area(self):
        main_frame = tk.LabelFrame(
            self.root, text="Defined Pose Lines", padx=10, pady=10
        )
        main_frame.pack(fill="both", expand=True, padx=10, pady=5)

        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill="x", pady=(0, 10))

        tk.Button(
            button_frame,
            text="Open / Load From File",
            command=self.open_sequence_file,
            width=20,
        ).pack(side="left", padx=5)

        self.run_all_button = tk.Button(
            button_frame, text="Run All", command=self.run_all, width=20
        )
        self.run_all_button.pack(side="left", padx=5)

        self.abort_button = tk.Button(
            button_frame,
            text="Abort Run All",
            command=self.abort_run_all,
            width=20,
            state="disabled",
        )
        self.abort_button.pack(side="left", padx=5)

        wait_frame = tk.Frame(main_frame)
        wait_frame.pack(fill="x", pady=(0, 10))

        tk.Label(wait_frame, text="Set Wait For All").pack(side="left", padx=(5, 5))

        self.global_wait_entry = tk.Entry(wait_frame, width=10)
        self.global_wait_entry.insert(0, "2")
        self.global_wait_entry.pack(side="left", padx=5)

        tk.Button(
            wait_frame,
            text="Apply To All Lines",
            command=self.apply_wait_to_all,
            width=18,
        ).pack(side="left", padx=5)

        # Scrollable area
        outer_rows_frame = tk.Frame(main_frame)
        outer_rows_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(outer_rows_frame, height=280)
        self.canvas.pack(side="left", fill="both", expand=True)

        scrollbar = tk.Scrollbar(
            outer_rows_frame, orient="vertical", command=self.canvas.yview
        )
        scrollbar.pack(side="right", fill="y")

        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.rows_frame = tk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window(
            (0, 0), window=self.rows_frame, anchor="nw"
        )

        self.rows_frame.bind("<Configure>", self.on_rows_frame_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)

        tk.Label(self.rows_frame, text="Line").grid(row=0, column=0, padx=5)
        tk.Label(self.rows_frame, text="Angles").grid(row=0, column=1, padx=5)
        tk.Label(self.rows_frame, text="Wait").grid(row=0, column=2, padx=5)
        tk.Label(self.rows_frame, text="Zero").grid(row=0, column=3, padx=5)
        tk.Label(self.rows_frame, text="Action").grid(row=0, column=4, padx=5)

    def on_rows_frame_configure(self, event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def load_default_rows(self):
        default_rows = [
            {"angles": [0, 0, 0, 0, 0, 0, 0, 0], "wait": 2, "zero": 180},
            {"angles": [45, 45, 45, 45, 45, 45, 45, 45], "wait": 2, "zero": 180},
        ]

        self.set_rows(default_rows)

    def clear_rows(self):
        for row in self.pose_rows:
            row["line_label"].destroy()
            row["angles_entry"].destroy()
            row["wait_entry"].destroy()
            row["zero_entry"].destroy()
            row["run_button"].destroy()

        self.pose_rows.clear()

    def set_rows(self, rows_data):
        self.clear_rows()

        for row_data in rows_data:
            self.add_pose_row(
                angles=row_data["angles"], wait=row_data["wait"], zero=row_data["zero"]
            )

        self.on_rows_frame_configure()

    def add_pose_row(self, angles, wait, zero):
        row_index = len(self.pose_rows)
        grid_row = row_index + 1

        line_label = tk.Label(self.rows_frame, text=str(row_index + 1))
        line_label.grid(row=grid_row, column=0, padx=5, pady=4)

        angles_entry = tk.Entry(self.rows_frame, width=75)
        angles_entry.insert(0, str(angles))
        angles_entry.grid(row=grid_row, column=1, padx=5, pady=4, sticky="ew")

        wait_entry = tk.Entry(self.rows_frame, width=8)
        wait_entry.insert(0, str(wait))
        wait_entry.grid(row=grid_row, column=2, padx=5, pady=4)

        zero_entry = tk.Entry(self.rows_frame, width=8)
        zero_entry.insert(0, str(zero))
        zero_entry.grid(row=grid_row, column=3, padx=5, pady=4)

        run_button = tk.Button(
            self.rows_frame, text="Run", command=lambda idx=row_index: self.run_one(idx)
        )
        run_button.grid(row=grid_row, column=4, padx=5, pady=4)

        self.pose_rows.append(
            {
                "line_label": line_label,
                "angles_entry": angles_entry,
                "wait_entry": wait_entry,
                "zero_entry": zero_entry,
                "run_button": run_button,
            }
        )

    def apply_wait_to_all(self):
        try:
            wait_value = float(self.global_wait_entry.get().strip())
        except Exception:
            messagebox.showerror("Input Error", "Please enter a valid wait value.")
            return

        for row in self.pose_rows:
            row["wait_entry"].delete(0, tk.END)
            row["wait_entry"].insert(0, str(wait_value))

        self.set_status(f"Applied wait={wait_value} to all lines", "blue")

    def set_run_all_state(self, running):
        self.is_running_all = running
        self.run_all_button.config(state="disabled" if running else "normal")
        self.abort_button.config(state="normal" if running else "disabled")

        for row in self.pose_rows:
            row["run_button"].config(state="disabled" if running else "normal")

    def abort_run_all(self):
        if self.is_running_all:
            self.abort_requested = True
            self.set_status("Abort requested... will stop after current line", "red")

    def open_sequence_file(self):
        script_folder = os.path.dirname(os.path.abspath(__file__))

        file_path = filedialog.askopenfilename(
            title="Open Sequence File",
            initialdir=script_folder,
            filetypes=[
                ("Text files", "*.txt"),
                ("Sequence files", "*.seq"),
                ("All files", "*.*"),
            ],
        )

        if not file_path:
            return

        try:
            rows_data = self.read_sequence_file(file_path)

            if not rows_data:
                messagebox.showerror(
                    "Error", "The selected file has no valid movement lines."
                )
                return

            self.set_rows(rows_data)
            self.set_status(f"Loaded file: {os.path.basename(file_path)}", "green")

        except Exception as e:
            messagebox.showerror("File Error", str(e))
            self.set_status("Failed to load file", "red")

    def read_sequence_file(self, file_path):
        rows_data = []

        with open(file_path, "r", encoding="utf-8") as file:
            for line_number, raw_line in enumerate(file, start=1):
                line = raw_line.strip()

                if not line:
                    continue

                if line.startswith("#"):
                    continue

                parts = line.split("|")

                if len(parts) != 3:
                    raise ValueError(
                        f"Invalid format on line {line_number}.\n"
                        f"Expected: [angles] | wait | zero"
                    )

                angles_text = parts[0].strip()
                wait_text = parts[1].strip()
                zero_text = parts[2].strip()

                angles = ast.literal_eval(angles_text)
                wait = float(wait_text)
                zero = float(zero_text)

                if not isinstance(angles, list):
                    raise ValueError(f"Angles must be a list on line {line_number}")

                if len(angles) == 0:
                    raise ValueError(f"Angles list is empty on line {line_number}")

                rows_data.append({"angles": angles, "wait": wait, "zero": zero})

        return rows_data

    def read_row(self, index):
        row = self.pose_rows[index]

        angles = ast.literal_eval(row["angles_entry"].get().strip())
        wait = float(row["wait_entry"].get().strip())
        zero = float(row["zero_entry"].get().strip())

        if not isinstance(angles, list):
            raise ValueError(f"Line {index + 1}: angles must be a list")

        return angles, wait, zero

    def read_all_rows(self):
        sequence = []

        for i in range(len(self.pose_rows)):
            angles, wait, zero = self.read_row(i)
            sequence.append({"angles": angles, "wait": wait, "zero": zero})

        return sequence

    def run_one(self, index):
        if not self.ensure_port():
            return

        try:
            angles, wait, zero = self.read_row(index)
        except Exception as e:
            messagebox.showerror("Input Error", str(e))
            return

        def worker():
            try:
                self.set_status_safe(f"Running line {index + 1}...", "blue")
                moveArray(self.port, angles, wait, zero)
                self.set_status_safe(f"Line {index + 1} done", "green")

            except Exception as e:
                self.set_status_safe(f"Line {index + 1} failed", "red")
                self.show_error("Error", str(e))

        self.run_thread(worker)

    def run_all(self):
        if not self.ensure_port():
            return

        if self.is_running_all:
            return

        try:
            sequence = self.read_all_rows()
        except Exception as e:
            messagebox.showerror("Input Error", str(e))
            return

        self.abort_requested = False
        self.set_run_all_state(True)

        def worker():
            try:
                self.set_status_safe("Running all lines...", "blue")

                for i, row in enumerate(sequence):
                    if self.abort_requested:
                        self.set_status_safe(
                            f"Run aborted before line {i + 1}", "red"
                        )
                        return

                    self.set_status_safe(f"Running line {i + 1}...", "blue")
                    moveArray(self.port, row["angles"], row["wait"], row["zero"])

                if self.abort_requested:
                    self.set_status_safe("Run aborted", "red")
                else:
                    self.set_status_safe("All lines done", "green")

            except Exception as e:
                self.set_status_safe("Run all failed", "red")
                self.show_error("Error", str(e))

            finally:
                self.root.after(0, lambda: self.set_run_all_state(False))

        self.run_thread(worker)


if __name__ == "__main__":
    root = tk.Tk()
    app = PoseGUI(root)
    root.mainloop()
