"""Independent camera view; closing it affects no acquisition or control state."""

import tkinter as tk


class CameraPopout:
    def __init__(self, parent, title, render, *, actions=(), status=lambda: '',
                 toolbar_builder=None, mirror_action_text=True, controls_enabled=lambda: True):
        self.window = tk.Toplevel(parent)
        self.window.title(title)
        self.window.geometry('1100x780')
        self.window.minsize(480, 320)
        self.render, self.status = render, status
        self.mirror_action_text, self.controls_enabled = mirror_action_text, controls_enabled
        self.after_id = None
        self.closed = False
        bar = tk.Frame(self.window)
        bar.pack(fill='x', padx=5, pady=3)
        self.extra_controls = tk.Frame(bar)
        self.extra_controls.pack(side='left')
        if toolbar_builder is not None:
            toolbar_builder(self.extra_controls)
        self.actions = []
        for label, command, source in actions:
            button = tk.Button(bar, text=label, command=lambda callback=command: callback() if self.controls_enabled() else None)
            button.pack(side='left', padx=2)
            self.actions.append((button, source))
        tk.Button(bar, text='Maximize', command=self.maximize).pack(side='right', padx=3)
        tk.Button(bar, text='Full screen', command=self.fullscreen).pack(side='right', padx=3)
        self.status_label = tk.Label(self.window, anchor='w', text='', wraplength=1000)
        self.status_label.pack(fill='x', padx=8)
        self.canvas = tk.Canvas(self.window, background='black', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Configure>', lambda event: self.refresh())
        self.window.bind('<Escape>', lambda event: self.window.attributes('-fullscreen', False))
        self.window.protocol('WM_DELETE_WINDOW', self.close)
        self.tick()

    @property
    def is_open(self):
        return not self.closed

    def show(self):
        self.window.deiconify()
        self.window.lift()

    def maximize(self):
        self.window.attributes('-fullscreen', False)
        try:
            self.window.state('zoomed')
        except tk.TclError:
            self.window.geometry(f'{self.window.winfo_screenwidth()}x{self.window.winfo_screenheight()}+0+0')

    def fullscreen(self):
        self.window.attributes('-fullscreen', True)

    def refresh(self):
        if self.closed:
            return
        self.render(self.canvas)
        self.status_label.config(text=self.status(), wraplength=max(400, self.window.winfo_width() - 16))
        for button, source in self.actions:
            state = source.cget('state') if source is not None else 'normal'
            button.config(state=state if self.controls_enabled() else 'disabled')
            if source is not None and self.mirror_action_text:
                button.config(text=source.cget('text'))
        for widget in self.extra_controls.winfo_children():
            if widget.winfo_class() in ('Entry', 'Checkbutton', 'Radiobutton', 'Button'):
                widget.config(state='normal' if self.controls_enabled() else 'disabled')

    def tick(self):
        self.after_id = None
        if self.closed:
            return
        self.refresh()
        self.after_id = self.window.after(100, self.tick)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.after_id is not None:
            self.window.after_cancel(self.after_id)
            self.after_id = None
        self.window.destroy()
