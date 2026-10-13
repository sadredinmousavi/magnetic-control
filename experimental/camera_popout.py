"""Independent camera view; closing it affects no acquisition or control state."""

import tkinter as tk


def show_view_dialog(owner, dialog, *args, **kwargs):
    """Keep errors/file dialogs attached to the camera view instead of the root."""
    views = [getattr(owner, 'camera_popout', None)]
    for name in ('calibration', 'commands'):
        tab = getattr(owner, name, None)
        if tab is not None:
            views.append(getattr(tab, 'camera_popout', None))
    popup = next((view for view in views if isinstance(view, CameraPopout) and view.is_open), None)
    parent = popup.window if popup is not None else getattr(owner, 'parent', getattr(owner, 'root', None))
    if parent is not None:
        kwargs.setdefault('parent', parent)
    try:
        return dialog(*args, **kwargs)
    finally:
        if popup is not None and popup.is_open:
            popup.show()


class CameraPopout:
    def __init__(self, parent, title, render, *, actions=(), status=lambda: '',
                 toolbar_builder=None, mirror_action_text=True, controls_enabled=lambda: True,
                 status_builder=None, status_refresh=None, render_version=None, on_close=None,
                 status_lines=0, action_captions=None, disabled_actions=()):
        self.window = tk.Toplevel(parent)
        self.window.title(title)
        self.window.geometry('1100x780')
        self.window.minsize(480, 320)
        self.window.attributes('-topmost', True)
        self.render, self.status = render, status
        self.mirror_action_text, self.controls_enabled = mirror_action_text, controls_enabled
        self.action_captions = action_captions or {}
        self.disabled_actions = set(disabled_actions)
        self.status_refresh = status_refresh
        self.render_version = render_version
        self.on_close = on_close
        self.last_render_key = None
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
        self.status_widgets = status_builder(self.window) if status_builder is not None else None
        self.status_label = tk.Label(self.window, anchor='nw', justify='left', text='', wraplength=1000,
                                     height=status_lines, width=1,
                                     background='#edf4ff', foreground='#173c68', padx=6, pady=3)
        self.status_label.pack(fill='x', padx=8)
        self.canvas = tk.Canvas(self.window, background='black', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Configure>', lambda event: self.refresh())
        self.window.bind('<Escape>', lambda event: self.window.attributes('-fullscreen', False))
        self.window.protocol('WM_DELETE_WINDOW', self.close)
        self.maximize()
        self.tick()

    @property
    def is_open(self):
        return not self.closed

    def show(self):
        self.window.deiconify()
        self.window.attributes('-topmost', True)
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
        key = None if self.render_version is None else (
            self.render_version(), self.canvas.winfo_width(), self.canvas.winfo_height())
        if key is None or key != self.last_render_key:
            self.render(self.canvas)
            self.last_render_key = key
        if self.status_refresh is not None:
            self.status_refresh(self.status_widgets)
        self.status_label.config(text=self.status(), wraplength=max(400, self.window.winfo_width() - 16))
        for button, source in self.actions:
            state = source.cget('state') if source is not None else 'normal'
            if button.cget('text') in self.disabled_actions:
                state = 'disabled'
            button.config(state=state if self.controls_enabled() else 'disabled')
            if source is not None and self.mirror_action_text:
                caption = source.cget('text')
                button.config(text=self.action_captions.get(caption, caption))
        for widget in self.extra_controls.winfo_children():
            if widget.winfo_class() in ('Entry', 'Spinbox', 'Checkbutton', 'Radiobutton', 'Button'):
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
        if self.on_close is not None:
            self.on_close()
