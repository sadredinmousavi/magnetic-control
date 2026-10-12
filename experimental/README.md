pip install dynamixel-controller
sudo apt-get install fswebcam
pip install opencv-python

GUI005 Commands tab
-------------------
Calibration and Commands both have a Pop-out camera button. The normal full pages
remain available, while a separate resizable window shows the same source and
overlays. Maximize or Full screen enlarges that view; Escape exits full screen.
Reopening the button raises the existing window instead of creating duplicates.
Calibration supports pin/ROI mouse placement and settled-robot Capture in the
pop-out; Commands supports target selection and synchronized Next/Finish/Stop.
Closing only the pop-out never stops tracking, recording or motor execution.
Closing GUI005 closes all camera windows and finalizes any recording.

Complete Center/+X/+Y calibration (or load a matching setup from `experimental/setups`)
and select the hardware case in Calibration. Select Live camera for command execution.
In Commands, load a file from `experimental/xy_sequences`:

`[x, y] | wait_seconds | motor_zero_degrees`

New exports also preserve equilibrium settings:
`[x, y] | wait_seconds | motor_zero_degrees | ratio | angle_deg`
Old three-column files default to ratio 1 and angle Auto. Angle is the equilibrium
orientation in robot XY, not the motor zero or calculated motor angles; ratio 1
ignores orientation. Select a point, edit Ratio / Equilibrium angle in the input
boxes and Apply to selected point. Alternatively double-click Ratio, Angle, Wait
or Zero in the table; Enter or leaving the cell applies, Escape cancels. Edits
are blocked during execution. Export XY sequence saves the edited points in
metres with explicit ratio/angle fields. Usage 2 exports these fields as well;
magnet-angle sequence files remain GUI001-compatible three-column files.

Calibration rectangle dimensions are in centimetres. Select sequence units
(`mm`, `cm`, or `m`) in Commands. `Auto` respects `# targets_xy_mm`,
`# targets_xy_cm`, or legacy `# targets_xy_m` headers, and uses millimetres
when no unit header exists. The table always displays millimetres; calculations
receive metres internally. Selecting different units reloads and reprojects the file.

Show all points fits the full image and complete desired path on screen, even
when some targets are outside the image or pin rectangle. Desired points are
cyan, beyond-pin points red, and the selected/current target orange. Select a
table row or click an image marker, then Run selected point, or choose Run all.
The preview reports sequence and pin extents in mm; the origin is the captured
robot Center, which need not equal the geometric rectangle center.
The table displays XY in millimetres. Commands use each row's ratio/angle and calculate fresh
motor angles for each point; the wait starts after sending that pose.
The row wait is a minimum motor hold. After it, tracking must confirm that every
expected robot is within Arrival tolerance (default 1 mm) continuously for the
In-range hold (default 0.5 seconds). Execution then waits for your Next point
button press; the last or single selected point uses Finish. There is no automatic
advance. Missing, stale or out-of-tolerance detections disable the button until
arrival is confirmed again. Set tolerance and hold before starting the run.
Run requires a live camera, connected controller, complete robot axes, and each
requested target inside the camera frame. Beyond-pin points require the explicit
Allow beyond pins option and a warning confirmation; their mapping is extrapolated
and you must verify that the target is physically safe. Running an inside selected
point is not blocked by other outside points. Calibration is locked while running.
Stop cancels unsent poses and interrupts waits; it does not undo the last pose.
Tracking gates progression; it does not recalculate corrective motor commands
while approaching a target. Simultaneous multi-target rows are not supported.

Phase 3 robot tracking starts when a complete calibration is applied, even before
loading a sequence. It uses the same dark-contrast detector, robot IDs and moving
ROI as Calibration. Set Robots to the actual count; adjust minimum area, contrast
and ROI radius in the command view. Tracking is not clipped to the pin rectangle,
so it can follow extrapolated targets. Partial/missing groups do not move the ROI
or produce valid group XY; current robot markers disappear when video is stale.
The view shows small green robot markers, actual group XY and active-target error
in mm. Desired lines, moving ROI and robot trails are hidden by default and can
be enabled separately. Export tracking CSV saves per-frame robot XY, group XY,
target XY, error, detection validity/counts, arrival readiness/error and moving ROI.
Stop tracking preserves measurements and pauses arrival progression without
undoing the motor pose; restart tracking to continue or use command Stop to end.

To export live video, apply calibration, start tracking, set Video FPS to match
the camera, then Start recording and choose a base name. Recording continues
across targets and tab switches until Stop / save recording (or closing GUI005).
It writes `_camera.mp4` (unmodified frame), `_annotated.mp4` (camera plus overlays),
`_overlay.mp4` (overlays on blue for chroma-key removal), and `_tracking.csv` with
recording frame IDs/timestamps and robot/target measurements. Overlay toggles
control desired lines, ROI and trails; robot markers and desired points remain.
Videos use source resolution, independent of preview zoom; odd frame dimensions
are padded by one pixel, not cropped. MP4 is constant FPS, so CSV timestamps are
the authority for variable-rate or dropped-frame timing. Existing recording names
are rejected to avoid overwriting. For CSV alone, use Export tracking CSV.




https://raspberrypi-guide.github.io/electronics/using-usb-webcams
https://tutorials-raspberrypi.com/raspberry-pi-security-camera-with-webcam/



```
python -m venv .venv
source .venv/bin/activate
which python

pip install -i https://mirror-pypi.runflare.com/simple dynamixel-sdk
```
