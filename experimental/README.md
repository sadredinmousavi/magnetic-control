pip install dynamixel-controller
sudo apt-get install fswebcam
pip install opencv-python

GUI005 Commands tab
-------------------
Connections has a Camera source selector: Pi SSH or Iriun / Local webcam.
For Iriun, install/start the Iriun desktop client and phone app, choose the webcam
source, set the local device index (try 0, 1, 2 if another camera appears), and
Connect webcam. Capture runs in a background thread using Windows DirectShow;
Requested resolution offers Auto (driver default), common presets and Custom
width/height. Presets are requests, not an enumerated list of guaranteed modes.
Disconnect/reconnect after choosing a size. The status always reports dimensions
of actual frames and warns in amber when the driver returns a different size.
No software upscaling is used. Resolution changes clear calibration/ROI; load a
setup matching actual dimensions or recalibrate. Both Phase 2 and Phase 3 consume
only the selected source's newest frames. Live processing FPS still controls
GUI/tracking throughput, not the webcam device frame rate.
Changing source/device cancels running control/precalculation, finalizes recording
and clears camera calibration/ROI even when resolutions match. New setups include
the source/device identity and reject a different one. Legacy Pi setups without
identity are not accepted for the new local-webcam source. Verify camera pose and
actual pin dimensions every time; reconnecting the same device is not proof it
has stayed physically aligned. Neither camera starts until you press Connect.

Calibration and Commands both have a Pop-out camera button. The normal full pages
remain available, while a separate resizable window shows the same source and
overlays. Maximize or Full screen enlarges that view; Escape exits full screen.
Reopening the button raises the existing window instead of creating duplicates.
Pop-out windows open maximized by default.
They stay on top until closed. Error, confirmation and file dialogs are owned by
the active camera view and restore it afterward instead of raising the main page.
Calibration supports pin/ROI mouse placement and settled-robot Capture in the
pop-out. Its single-row toolbar includes 4 pins / ROI modes, Manual pins,
pin/robot ROI radius inputs, Center/+X/+Y moves, Capture, Reset axes, Clear pins,
Clear ROI. These share the main page's state and settings.
Calibration pop-out controls are locked during a Phase 3 run. Commands supports
target selection and synchronized Next/Finish/Stop.
Closing only the pop-out never stops tracking, recording or motor execution.
While a pop-out is open its embedded camera preview stops drawing and displays
a placeholder. Closing it immediately restores the embedded view. Tracking and
recording keep running; the Connections preview also skips drawing while a
pop-out is open. Tracking processes native-resolution pixels only around the
moving ROI with a 32-pixel blur halo; detections are translated back to full-frame
coordinates. Pin clipping, identity checks and physical XY mapping are unchanged.
Closing GUI005 closes all camera windows and finalizes any recording.
Press Prepare axis moves beside Hardware case (or Prepare in the pop-out) once.
It computes seven poses: Center, +X/+2X/+3X and +Y/+2Y/+3Y (10/20/30 mm from
Center along the selected axis), in the background without connecting or moving
motors. Afterwards these moves reuse the prepared ratio-1 poses; Phase 3 continues
to synthesize its own per-point settings freshly. Changing the selected case or
its hardware model requires preparation again. Calibration has no settle-time
input or timed delay: Capture enables after the command is sent. Visually wait
until the robot settles, then press Capture; sending a command does not prove
the motors or robot have reached their positions. Fresh-frame checks still apply.
Moves can be repeated in any order once the current move finishes; a new move
only requires selected/prepared hardware and a connected serial controller, not
pins, ROI, a camera or a robot detection. The command uses physical case XY, not
image coordinates. Capture still requires fresh live frames, four pins, a valid
ROI and exactly one unambiguously tracked robot.
Each new move
replaces an uncaptured pending position without silently saving it. For axes,
capture Center first, then any X distance and any Y distance. Only the latest
successful capture for each axis is kept: +3X replaces the earlier +X/+2X
reference, and likewise for Y. A move alone never replaces a saved capture.
Setup files retain both the chosen move/distance and the captured pixel position;
older setups using 10 mm moves remain loadable. Four-pin dimensions determine
physical scale; the captures determine origin and axis directions, so a command
position error is not imposed as a new pixel-to-mm scale. Main and pop-out views
have the same compact buttons, with blue X and green Y groups and hover hints.
Prepare again after updating to build all seven poses.
Capturing a new Center resets dependent X/Y
samples, but merely testing a Center move does not erase saved axes.
Calibration is fixed to one robot. Placing an ROI automatically starts detection
and following, without requiring pins. Without four pins, physical XY fields are
blank and pixel positions remain available. The same tracker supplies axis Capture;
no second detector runs per frame. Clear ROI stops detection. Pins and fresh
unambiguous observations are still required to save axis captures.
Phase 3 has its own Phase 3 robots input (default 1). Enter the experimental
robot count there; applying/loading Phase 2 calibration does not overwrite it.
Both views show color-coded steps: 1. 4 pins,
2. ROI and robot, 3. Axis (green completed, blue active, gray waiting). Capture
is initially disabled; send a Center/+X/+Y command first.
It enables for the pending position, then disables after a successful
capture. The status hint explains waiting, ready, completed or Phase 3 lock states.

Calibration is live-camera only: connect the Pi camera in Connections. Movie
loading, playback, rewind and movie timers have been removed from GUI005;
There is no Freeze / Resume: pin/ROI placement also stays live. GUI004 post-processing is unchanged.
Live processing FPS defaults to 5 (editable 1–15); each cycle drains the frame
queue and processes only the latest image. It preserves full resolution and
does not change the Pi's capture FPS. Hidden previews skip drawing, unchanged
pop-out images skip redraws, and both Calibration views share detection-mask
preview results. Tracking/capture still operate on current frames.
Calibration resizes its preview first, then draws thin overlays only at display
resolution. Native detection and exported videos are unaffected. Pure detection
and the two tabs share read-only camera buffers instead of copying full frames.
Automatic axis tracking skips hidden idle views, but always runs for a pending
move/capture or open Calibration pop-out. Explicitly started tracking and CSV
logging are retained, including when changing tabs.
Complete Center/+X/+Y calibration (or load a matching setup from `experimental/setups`)
and select the hardware case in Calibration.
Commands includes Precalculate all only on the main page. It
computes every row's XY/ratio/equilibrium-angle pose in the background without
motor IO, then Run selected / Run all reuse the finished bundle. Duplicate
identical targets/settings are solved once. Sequence/case changes require a new
precalculation, and the hardware model is checked again before every send.
Partial or cancelled results are not used. Stop cancels after the current solver
returns; calibration controls are locked while preparing. Without precalculation,
commands retain fresh per-point synthesis. Arrival, manual Next, waits and motor
zero settings apply in either mode.
The Commands pop-out uses grouped controls: First, Selected, Next/Finish, Manual, Control,
Stop. First starts the sequence at point 1, then waits for your Next
press at each step; it is disabled during execution/precalculation. Control is
an opt-in force PID controller, disabled by default. Default Kp/Ki/Kv are
0.015/0.00015/0.015 in the GUI's micro-newton/mm/second units (Ki=0 selects PD):
position feedback and filtered-velocity damping allocate total magnetic force
during motion at the measured position. Disturbance learning and the stationary-only
controller have been removed. Positive Ki enables bounded integral correction
with anti-windup. The bounded integral force now carries between successive
points and matching selected runs. The first pose for the next point already
includes that force bias, allocated using the new baseline's stiffness; no
uncorrected baseline is sent first. P/D transients and old motor angles are not
accumulated. Reset PID memory (while stopped) starts fresh without moving motors.
Changing calibration, hardware, source, robot count, path or serial connection
invalidates memory. Memory is not persisted across application restarts. At the
new defaults, Kp/Ki is 100 seconds: first-time bias learning is deliberately slow.
Hold speed (0.5 mm/s) only holds the motor pose when also inside
arrival tolerance; it never triggers a learned-force update. Row Wait delays Next
navigation, not feedback or braking.
It preserves baseline trap stiffness and bounds angle changes. Main/pop-out
details show error, integral, requested/allocated force, residual and calculated/sent
angles. Preview only computes without sending corrections. The former PID,
XY Alpha and adaptive-limit settings have been removed from GUI005; the new
PID gains have physical force units and configurable cadence/filtering.
See [POSITION_CONTROL.md](POSITION_CONTROL.md) for the algorithm and diagnostics.
In Commands, load a file from `experimental/xy_sequences`:

Commands uses the loaded/applied calibration ROI, area and contrast by default;
tracking starts automatically. Set Phase 3 robots independently. Reposition it
with Place ROI if needed; loading another setup restores its defaults automatically
when entering Commands. There is no Use current calibration button. Without calibration, choose Place ROI and click the
camera around the group. This is a separate moving ROI from Phase 2 and
works without a sequence or completed calibration. Pixel-only observations leave
physical XY blank. Apply a complete Phase 2 calibration before commanding points.
Place ROI / Clear ROI also work in the camera pop-out. Its synchronized Robots,
ROI px (radius), Area
and Contrast inputs, plus the ROI visibility toggle. These inputs are locked
during execution/precalculation; Next/Stop remain available.
Clearing the ROI stops tracking and cancels active execution; replacing it
requires an idle run. Camera resolution changes require
placing it again. The large instruction paragraph is now hover help on Help ?.

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
The row wait is a minimum motor hold. Next can queue during calculation,
without waiting for arrival. An early press is queued until the minimum
hold finishes. The last or single selected point uses Finish. There is no automatic
advance or queued permission for subsequent rows. Tracking tolerance/hold remain
arrival indicators and force-control position tolerance, not permission to advance. Missing
robot detection pauses corrections but does not revoke manual Next. Camera loss,
controller disconnection and Stop still cancel execution.
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
Live processing FPS (default 5), then Start recording and choose a base name. Recording continues
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
