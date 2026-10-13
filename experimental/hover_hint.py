"""Small delayed, nonmodal hover help for crowded camera controls."""
import tkinter as tk


class HoverHint:
    def __init__(self, widget, text):
        self.widget, self.text = widget, text
        self.timer, self.window = None, None
        widget.bind('<Enter>', self.schedule, add='+')
        widget.bind('<Leave>', self.hide, add='+')
        widget.bind('<ButtonPress>', self.hide, add='+')
        widget.bind('<Destroy>', self.hide, add='+')

    def schedule(self, event=None):
        self.hide()
        self.timer = self.widget.after(400, self.show)

    def show(self):
        self.timer = None
        self.window = tk.Toplevel(self.widget)
        self.window.overrideredirect(True)
        self.window.attributes('-topmost', True)
        self.window.geometry(f'+{self.widget.winfo_rootx()}+{self.widget.winfo_rooty() + self.widget.winfo_height() + 6}')
        tk.Label(self.window, text=self.text, justify='left', wraplength=520, background='#fff5cf',
                 foreground='#252525', relief='solid', borderwidth=1, padx=10, pady=8).pack()

    def hide(self, event=None):
        if self.timer is not None:
            self.widget.after_cancel(self.timer)
            self.timer = None
        if self.window is not None:
            self.window.destroy()
            self.window = None
