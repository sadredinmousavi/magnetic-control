"""Offline recording preview and export, independent of live camera controls."""
from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, ttk

import cv2
from PIL import Image, ImageTk

from command_recording import RecordingPreview, annotate_recording, video_crop_circle, render_time_annotation
from command_tab import CommandTab


class RecordingExportTab:
    def __init__(self, parent, *, operation_allowed=lambda: True):
        self.parent, self.operation_allowed = parent, operation_allowed
        self.busy = self.closed = False
        self.thread = self.after_id = self.recording = self.preview_image = None
        self.cancel_event = threading.Event()
        self.frame_var = tk.DoubleVar(value=0.)
        self.start_frame_var = tk.StringVar(value='0')
        self.marker_var = tk.StringVar(value='Rectangle')
        self.crop_var = tk.BooleanVar(value=True)
        self.circle_crop_var = tk.BooleanVar(value=False)
        self.crop_radius_var = tk.StringVar(value='300')
        self.crop_center = None
        self.crop_placing = False
        self.preview_transform = None
        self.time_position = (.03, .03)
        self.time_bounds = self.time_drag_offset = None
        self.preview_base = None
        self.preview_elapsed = 0.
        header = tk.Frame(parent)
        header.pack(fill='x', padx=8, pady=5)
        self.load_button = tk.Button(header, text='Load recording', command=self.load_recording)
        self.load_button.pack(side='left')
        self.file_label = tk.Label(header, text='Select *_camera.mp4; keep its CSV and metadata beside it.', anchor='w', width=1)
        self.file_label.pack(side='left', fill='x', expand=True, padx=8)
        panel = tk.LabelFrame(parent, text='Choose annotations — independent of the live tabs')
        panel.pack(fill='x', padx=8, pady=2)
        controls = tk.Frame(panel)
        controls.pack(fill='x')
        self.options, self.option_widgets = {}, []
        for label, key, default in (
                ('Desired lines', 'show_path', True), ('Desired points', 'show_points', True),
                ('Moving ROI', 'show_roi', True), ('Calibration boundary', 'show_pins', True),
                ('Robot markers', 'show_robots', True), ('Robot trails', 'show_trails', False),
                ('Center trail', 'show_center_trail', True), ('Group center', 'show_group_center', True),
                ('Spread axes', 'show_spread_axes', False), ('Cargo outline', 'show_cargo', True),
                ('Commanded point', 'show_commanded', True), ('Waypoint', 'show_waypoint', True),
                ('Text labels', 'show_labels', False), ('Elapsed time (drag in preview)', 'show_time', False)):
            variable = self.options[key] = tk.BooleanVar(value=default)
            self.option_widgets.append(tk.Checkbutton(controls, text=label, variable=variable,
                                                     command=self.preview, pady=0))
        crop = tk.Checkbutton(controls, text='Apply saved crop', variable=self.crop_var, command=self.preview, pady=0)
        self.option_widgets.append(crop)
        marker_cell = tk.Frame(controls)
        tk.Label(marker_cell, text='Robot marker').pack(side='left')
        self.marker = ttk.Combobox(marker_cell, textvariable=self.marker_var,
                                   values=('Rectangle', 'Cross'), state='readonly', width=10)
        self.marker.pack(side='left', padx=3)
        self.marker.bind('<<ComboboxSelected>>', lambda event: self.preview())
        CommandTab.wrap_annotation_controls(controls, [*self.option_widgets, marker_cell])
        crop_bar = tk.Frame(parent)
        crop_bar.pack(fill='x', padx=8, pady=2)
        circle_check = tk.Checkbutton(crop_bar, text='Circle crop (auto-fit 4 pins)',
                                     variable=self.circle_crop_var, command=self.change_crop)
        circle_check.pack(side='left')
        self.option_widgets.append(circle_check)
        tk.Label(crop_bar, text='Without pins: radius px').pack(side='left')
        self.crop_radius_entry = tk.Entry(crop_bar, textvariable=self.crop_radius_var, width=6)
        self.crop_radius_entry.pack(side='left', padx=4)
        self.crop_radius_entry.bind('<Return>', lambda event: self.change_crop())
        self.crop_place_button = tk.Button(crop_bar, text='Place crop', command=self.begin_crop_placement)
        self.crop_place_button.pack(side='left')
        bar = tk.Frame(parent)
        bar.pack(fill='x', padx=8, pady=3)
        self.preview_button = tk.Button(bar, text='Preview', command=self.preview, state='disabled')
        self.preview_button.pack(side='left')
        self.export_button = tk.Button(bar, text='Export / Save video', command=self.export_video, state='disabled')
        self.export_button.pack(side='left', padx=4)
        self.save_image_button = tk.Button(bar, text='Save preview PNG', command=self.save_preview, state='disabled')
        self.save_image_button.pack(side='left')
        self.cancel_button = tk.Button(bar, text='Cancel', command=self.cancel_event.set, state='disabled')
        self.cancel_button.pack(side='left', padx=4)
        self.frame_label = tk.Label(bar, text='No recording loaded', anchor='w')
        self.frame_label.pack(side='left', padx=8)
        self.slider = ttk.Scale(parent, from_=0, to=1, variable=self.frame_var, state='disabled',
                                 command=lambda value: self.update_frame_label())
        self.slider.pack(fill='x', padx=10)
        self.slider.bind('<ButtonRelease-1>', lambda event: self.preview())
        self.slider.bind('<KeyRelease>', lambda event: self.preview())
        trim_bar = tk.Frame(parent)
        trim_bar.pack(fill='x', padx=8, pady=3)
        tk.Label(trim_bar, text='Export start frame (0-based):').pack(side='left')
        self.start_frame_entry = tk.Spinbox(trim_bar, from_=0, to=0, width=8,
                                           textvariable=self.start_frame_var, state='disabled')
        self.start_frame_entry.pack(side='left', padx=4)
        self.start_frame_entry.bind('<Return>', lambda event: self.preview())
        self.start_apply_button = tk.Button(trim_bar, text='Apply start', command=self.preview, state='disabled')
        self.start_apply_button.pack(side='left')
        self.start_current_button = tk.Button(trim_bar, text='Use current frame as start',
                                              command=self.use_current_as_start, state='disabled')
        self.start_current_button.pack(side='left', padx=4)
        tk.Label(trim_bar, text='Earlier frames are removed; time starts at 00.0 s.').pack(side='left', padx=4)
        self.status = tk.Label(parent, text='Load a recording, select overlays, preview, then export.',
                               anchor='w', width=1, background='#edf4ff')
        self.status.pack(fill='x', padx=8, pady=3)
        self.canvas = tk.Canvas(parent, background='black', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True, padx=8, pady=(0, 6))
        self.canvas.bind('<Configure>', lambda event: self.draw_preview())
        self.canvas.bind('<Button-1>', self.begin_preview_drag)
        self.canvas.bind('<B1-Motion>', self.drag_time)
        self.canvas.bind('<ButtonRelease-1>', self.end_preview_drag)

    def begin_preview_drag(self, event):
        if self.crop_placing:
            self.place_crop(event)
            return
        if self.busy or self.preview_transform is None or self.time_bounds is None:
            return
        left, top, scale = self.preview_transform
        x, y = (event.x-left)/scale, (event.y-top)/scale
        bx, by, width, height = self.time_bounds
        if bx <= x <= bx+width and by <= y <= by+height:
            self.time_drag_offset = (x-bx, y-by)
            self.canvas.config(cursor='fleur')

    def drag_time(self, event):
        if self.time_drag_offset is None or self.busy:
            return
        left, top, scale = self.preview_transform
        dx, dy = self.time_drag_offset
        height, width = self.preview_base.shape[:2]
        _, _, bw, bh = self.time_bounds
        x = max(0, min(width-bw, (event.x-left)/scale-dx))
        y = max(0, min(height-bh, (event.y-top)/scale-dy))
        self.time_position = (x/width, y/height)
        self.draw_preview()

    def end_preview_drag(self, event):
        if self.time_drag_offset is not None:
            self.drag_time(event)
            self.time_drag_offset = None
            self.canvas.config(cursor='')

    def crop_corners(self):
        if self.recording is None:
            return None
        snapshot = self.recording.observations[0][0]
        setup = snapshot.get('setup') or self.recording.info.get('calibration') or {}
        corners = setup.get('corners_px')
        return corners if corners is not None and len(corners) == 4 else None

    def selected_crop(self):
        if self.crop_placing:
            return False, None
        if not self.circle_crop_var.get():
            return self.crop_var.get(), None
        corners = self.crop_corners()
        circle = video_crop_circle(tuple(self.recording.info['source_size_px']),
            corners=corners, fit=corners is not None, center=self.crop_center,
            radius=float(self.crop_radius_var.get()))
        return True, circle

    def change_crop(self):
        self.crop_placing = False
        self.canvas.config(cursor='')
        self.preview()

    def begin_crop_placement(self):
        if self.recording is None or self.busy:
            return
        self.circle_crop_var.set(True)
        if self.crop_corners() is not None:
            self.change_crop()
            return
        self.crop_placing = True
        self.canvas.config(cursor='crosshair')
        self.preview()

    def place_crop(self, event):
        if not self.crop_placing or self.busy or self.preview_transform is None:
            return
        left, top, scale = self.preview_transform
        x, y = (event.x-left)/scale, (event.y-top)/scale
        width, height = self.recording.info['source_size_px']
        if 0 <= x < width and 0 <= y < height:
            self.crop_center = (x, y)
            self.change_crop()

    def selected_options(self):
        return dict({key: variable.get() for key, variable in self.options.items()},
                    robot_marker=self.marker_var.get(), time_position=self.time_position)

    def selected_start_frame(self):
        try:
            index = int(self.start_frame_var.get())
        except ValueError:
            raise ValueError('Export start frame must be a whole frame number.') from None
        self.recording.time_origin(index)  # Validate against the loaded recording.
        return index

    def use_current_as_start(self):
        if self.recording is None or self.busy:
            return
        self.start_frame_var.set(str(round(self.frame_var.get())))
        self.preview()

    def update_frame_label(self):
        if self.recording is not None:
            index = max(0, min(len(self.recording.frames)-1, round(self.frame_var.get())))
            _, stamp = self.recording.observations[self.recording.frames[index]]
            elapsed = stamp-float(self.recording.info['started_monotonic_s'])
            self.frame_label.config(text=f'Frame {index}/{len(self.recording.frames)-1} · capture {elapsed:.3f} s')

    def set_busy(self, busy):
        self.busy = busy
        self.load_button.config(state='disabled' if busy else 'normal')
        for widget in self.option_widgets:
            widget.config(state='disabled' if busy else 'normal')
        self.marker.config(state='disabled' if busy else 'readonly')
        self.crop_radius_entry.config(state='disabled' if busy else 'normal')
        loaded = not busy and self.recording is not None
        for widget in (self.start_frame_entry, self.start_apply_button, self.start_current_button):
            widget.config(state='normal' if loaded else 'disabled')
        self.crop_place_button.config(state='normal' if loaded else 'disabled')
        self.slider.config(state='normal' if loaded else 'disabled')
        self.preview_button.config(state='normal' if loaded else 'disabled')
        self.export_button.config(state='normal' if loaded and not self.crop_placing else 'disabled')
        self.save_image_button.config(state='normal' if loaded and self.preview_image is not None and not self.crop_placing else 'disabled')
        self.cancel_button.config(state='normal' if busy else 'disabled')

    def start_task(self, message, work, done):
        if self.busy or self.closed:
            return
        if not self.operation_allowed():
            self.status.config(text='Stop control and finalize recording before loading, previewing or exporting.')
            return
        self.cancel_event = threading.Event()
        self.cancel_button.config(command=self.cancel_event.set)
        self.set_busy(True)
        self.status.config(text=message)
        result = {}
        def run():
            try:
                result['value'] = work(self.cancel_event.is_set, lambda count: result.update(progress=count))
            except Exception as error:
                result['error'] = str(error)
        self.thread = threading.Thread(target=run, daemon=True, name='gui005-recording-export')
        self.thread.start()
        def poll():
            self.after_id = None
            if self.closed:
                return
            if self.thread.is_alive():
                suffix = f' {result["progress"]} frames' if 'progress' in result else ''
                self.status.config(text=message+suffix)
                self.after_id = self.parent.after(100, poll)
                return
            self.set_busy(False)
            if 'error' in result:
                self.status.config(text=result['error'])
            else:
                done(result['value'])
        poll()

    def load_recording(self):
        if self.busy:
            return
        path = filedialog.askopenfilename(parent=self.parent, title='Load camera recording',
                                         filetypes=[('Camera recording', '*_camera.mp4')])
        if not path:
            return
        def loaded(recording):
            self.recording = recording
            self.crop_center = None
            self.crop_placing = False
            self.canvas.config(cursor='')
            self.preview_image = None
            self.preview_base = None
            self.time_bounds = self.time_drag_offset = None
            self.time_position = (.03, .03)
            self.canvas.delete('all')
            self.file_label.config(text=str(recording.video_path))
            self.frame_var.set(0.)
            self.start_frame_var.set('0')
            self.start_frame_entry.config(to=len(recording.frames)-1)
            self.slider.config(from_=0, to=len(recording.frames)-1)
            self.set_busy(False)
            self.preview()
        self.start_task('Loading recording…', lambda cancel, progress: RecordingPreview(path, cancelled=cancel), loaded)

    def preview(self):
        if self.recording is None or self.busy:
            return
        recording, index = self.recording, round(self.frame_var.get())
        options = self.selected_options()
        options['show_time'] = False  # Composite locally for smooth mouse dragging.
        try:
            start_frame = self.selected_start_frame()
            crop, circle = self.selected_crop()
        except (ValueError, TypeError) as error:
            self.status.config(text=str(error))
            return
        index = max(start_frame, min(len(recording.frames)-1, index))
        self.slider.config(from_=start_frame)
        self.frame_var.set(index)
        def shown(value):
            self.preview_base, elapsed = value
            self.preview_elapsed = elapsed
            self.draw_preview()
            self.update_frame_label()
            self.save_image_button.config(state='disabled' if self.crop_placing else 'normal')
            self.status.config(text=f'Preview ready · frame {index} · export time {elapsed:.3f} s. Export uses these same options.')
            if self.crop_placing:
                self.status.config(text='Click the preview to place the circle center; radius uses original camera pixels.')
        self.start_task('Rendering preview…',
            lambda cancel, progress: recording.render(index, options=options, apply_crop=crop, crop_circle=circle,
                                                       start_frame=start_frame, cancelled=cancel), shown)

    def draw_preview(self):
        if self.preview_base is None or self.closed:
            return
        self.preview_image = self.preview_base.copy()
        self.time_bounds = render_time_annotation(self.preview_image, self.preview_elapsed, self.selected_options())
        image = self.preview_image
        width, height = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        scale = min(width/image.shape[1], height/image.shape[0])
        resized = cv2.resize(image, (max(1, round(image.shape[1]*scale)), max(1, round(image.shape[0]*scale))))
        self.preview_transform = ((width-resized.shape[1])/2, (height-resized.shape[0])/2, scale)
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
        self.canvas.delete('all')
        self.canvas.create_image(width/2, height/2, image=self.photo, anchor='center')

    def export_video(self):
        if self.recording is None or self.busy or self.crop_placing:
            return
        try:
            start_frame = self.selected_start_frame()
            crop, circle = self.selected_crop()
        except (ValueError, TypeError) as error:
            self.status.config(text=str(error))
            return
        source = self.recording.video_path
        path = filedialog.asksaveasfilename(parent=self.parent, title='Save annotated video',
            initialdir=str(source.parent), initialfile=source.stem.removesuffix('_camera')+'_annotated.mp4',
            defaultextension='.mp4', filetypes=[('Video', '*.mp4')])
        if not path:
            return
        options = self.selected_options()
        self.start_task('Exporting video…',
            lambda cancel, progress: annotate_recording(source, path, options=options, apply_crop=crop, crop_circle=circle,
                                                       start_frame=start_frame, cancelled=cancel, progress=progress),
            lambda count: self.status.config(text=f'Saved {count} annotated frames: {path}'))

    def save_preview(self):
        if self.preview_image is None or self.busy:
            return
        path = filedialog.asksaveasfilename(parent=self.parent, title='Save preview image',
                                           defaultextension='.png', filetypes=[('PNG image', '*.png')])
        if path:
            try:
                if not cv2.imwrite(path, self.preview_image):
                    raise ValueError('Could not save preview image.')
                self.status.config(text=f'Saved preview: {path}')
            except (OSError, ValueError, cv2.error) as error:
                self.status.config(text=str(error))

    def close(self):
        self.closed = True
        self.cancel_event.set()
        if self.after_id is not None:
            self.parent.after_cancel(self.after_id)
        if self.thread is not None:
            self.thread.join(timeout=2.)
