# Commands position control

Control is optional and off by default. Enabling it requires confirmation. It
does not move anything until a sequence runs. The existing Run confirmation,
camera/controller checks, arrival hold, Next/Finish and Stop remain in effect.

## Idea

In calibrated robot XY (millimetres):

`error = desired - measured`

`u_candidate = u_last_sent + Alpha*error`

`commanded = desired + bounded(u_candidate)`

The desired sequence point never changes. This is incremental, discrete
integral-like compensation, NOT `u = Kp*error`. It retains the bias needed to
hold the robot at the target. **Only a successful motor send commits u**;
rejected/stale calculations never accumulate into the next correction.
GUI005 uses Alpha (default 0.2); Ki/Kd inputs are removed from this tab. The
reusable PositionPID class retains its legacy PID mode for other callers.
There is a vector correction limit and a slew-rate limit; the starting
correction radius is 2 mm. GUI defaults are slew 5 mm/s and minimum update period
0.25 s (the reusable PID class retains its original 1 mm/s and 1 s defaults).
These are starting settings, not experimentally tuned gains. Test on your
hardware carefully. Settings are locked during a run.

## Adaptive Alpha (formerly Adaptive Kp)

The GUI's **Adaptive Alpha** checkbox defaults on; Control itself still defaults
off. Uncheck it for fixed Alpha (still incremental control). The existing **Arrival tolerance
(mm)** is also the feedback goal, default **1 mm**. Starting Alpha defaults to 0.2;
adaptive Alpha is bounded between 0.05 and the editable **Alpha max** (default 1).
Current Alpha and the adjustment reason appear in both
control-detail views and tracking/recording CSVs.

This is a conservative error-based gain scheduler, not a plant-identification
or guaranteed-convergence controller. It consumes only fresh, complete-group
observations following successfully SENT corrections, at least one control
period after the send. Each response with less than 5% (or 0.05 mm)
improvement increases Alpha immediately: **Adapt %** defaults to 50% per response
far from the target. The increase tapers toward 10% near tolerance (at the
default pace); configurable Adapt % must be greater than zero and at most 100.
For example, stagnant 10 mm error with 1 mm tolerance raises Alpha from
0.2 → 0.3 → 0.45 → 0.675 → 1 in four eligible responses, rather than four checks
for each 10% step. Crossing the target (error-vector reversal) or
error growth greater than 10% (or 0.2 mm) reduces Alpha by 20%. Clear improvement
holds Alpha. Correction/slew saturation suppresses increases. These thresholds are
initial heuristics, not hardware-tuned values.

Within tolerance, corrections pause using the existing arrival logic; gains do
not keep increasing. For multiple robots this controls the group centroid;
individual arrival is still reported separately. Tracking loss, Control toggles
and each new point reset Alpha to the starting value and clear response history,
but preserve that point's last sent u. Inside tolerance u is held, not zeroed.
No new motor-command path is introduced: workspace, freshness, Stop, correction
and slew limits still apply. Adaptive Alpha cannot fix bad calibration, tracking
noise, a stuck robot or insufficient correction range; reaching 1 mm is not
guaranteed. Begin with one robot and conservative limits.

## Adaptive correction limit

**Adaptive limit** defaults on in the GUI, independently of Adaptive Alpha.
**Limit mm** is now the starting correction radius (default 2 mm); **Limit max
(mm)** is the editable ceiling (default 20 mm, not a hardware-tuned value).
Uncheck Adaptive limit to retain the original fixed cap.

After each valid applied response at the actual correction cap with error
still outside tolerance and without substantial worsening, the radius expands
by half of Adapt % (default 25%) or 0.25 mm, whichever is larger, up to the ceiling. A slew-limited command
that has not reached the cap does not count. Overshoot/error growth reduces the
radius by 20%, but never below the starting radius or the magnitude of the
current applied output, avoiding abrupt clipping. Reaching tolerance holds the
radius. No successfully sent response means no expansion. New sends preserve
an unconsumed response anchor so camera/solver timing cannot indefinitely
postpone adaptation. Reset conditions also restore the starting radius.

Alpha remains held while capped; once the radius opens enough, gain adaptation can
resume. The existing slew limit remains unchanged and every corrected target
must still be inside the camera and, unless extrapolation is enabled, the pins. Blocked commands cannot drive limit
growth. Current radius and its adjustment reason appear in both detail views
and CSVs. Incremental compensation is bounded by this radius and does not
guarantee 1 mm accuracy.

There is no added settling sleep between corrections. Updates require a new
valid camera sample, speed-based settling, and the configured minimum period. One motor-angle solve
and send runs at a time; actual cadence also depends on live FPS, solver runtime
and serial throughput. The 0.25 s setting is not a guaranteed 4 Hz send rate.
The initial sequence row's Wait still applies before feedback begins. A new
incremental update also needs a sample at least one period AFTER the previous
successful send; repeated/pre-send observations cannot repeatedly accumulate u.

## Speed-based settling before correction

GUI005 now holds u while the robot is moving. Each successfully sent command
(including the initial learned replay command) starts a new settling cycle.
Speed is estimated from calibrated XY samples in mm using a rolling linear fit
over **Window s**, with at least three fresh frames. A residual-motion check
also detects oscillation that would otherwise cancel out in the fitted velocity.
For multiple robots, identities are ordered and the maximum individual speed is
used; opposite motions cannot cancel through the group centroid.

The default editable settings are:

- Speed mm/s: **0.5**, maximum speed for every robot.
- Window s: **0.5**, velocity estimation interval.
- Hold s: **0.5**, continuous observed time below that speed.
- Min wait s: **0.5**, minimum time after sending before the low-speed hold starts.

Only after the speed window and hold are satisfied does the controller compute
error from the mean robot positions during the low-speed hold, update Alpha/the
adaptive limit, and propose ONE incremental correction. Sending it resets the
settling window. In tolerance, the learned u remains applied. Settling means
stationary; arrival additionally requires the desired-position tolerance. The
settings are locked during a run. These defaults require experimental tuning
to camera noise and robot response; minimum wait should cover delayed motion.

Repeated frames cannot advance the hold. Tracking loss, an observation gap over
one second, or a changed robot count discard the settling history. Stale data
never count as settled. If motion continues, corrections stay paused and the
display stays Moving; there is no timeout that forces a correction. Next/Stop
remain available. Before a calculated pose is sent, settling is checked again;
new motion during calculation causes the pose to be discarded.

Both camera views show maximum speed, threshold, hold progress, and controller
state (Minimum wait / Measuring speed / Moving / Settling / Reached). CSV adds
robot_speed_mm_s, settling_state, settle_hold_elapsed_s, settled_xy_mm and
settled. Learned corrections, recording crop and the motor solver are retained.

## Learning a path and replaying it

Each sequence row has its own u_last in mm (including repeated desired points).
After an actual corrective send, GUI005 remembers that row's applied u. **Use
learned u** defaults on: restarting with First/Run all initially commands
`p_d[i] + u_last[i]`, even with Control off. Arrival and desired-point overlays
still refer to the ORIGINAL p_d. Missing/untrained rows start with zero u.
Control on during replay refines the existing u for changed conditions. Control
off holds the last sent pose and disables new corrections, not learned replay.

Corrections remain in GUI memory across runs. **Save u** writes a dated JSON
file in `experimental/setups`; **Load u** restores it after restarting the GUI.
Files include desired XY, u_last, commanded XY, ratio and angle. This is explicit
save/load, not automatic disk overwriting. **Clear u** clears only the current
path/setup in memory; files remain. Unchecking Use learned u starts a fresh run
at nominal points (Control can learn replacements).

A context fingerprint includes the full sequence/order/settings, camera and
resolution, four pins and physical dimensions, captured axes, hardware model,
and Phase 3 robot count. Changed contexts never silently reuse old corrections;
loading a mismatched file is rejected. Moving the detection ROI does not
invalidate physical calibration or learned u. Saved corrections are validated
against current limits, the camera and the selected pin-extrapolation setting before moving.
Matching context does not prove that experimental conditions are unchanged.

The Commands table shows Desired X/Y, u X/Y and Command X/Y, all in mm.
Each new context starts with u = (0, 0); Command XY = Desired XY + u.
After a correction is successfully sent, its point's u and Command XY update
in the table. Calculated proposals that are blocked or never sent do not replace u.
The last sent u stays in memory for replay, including when restarting from the first point.
**Save u** remains available during a run in both camera views: press it when
satisfied with the settled position to save a snapshot of all points to a setup JSON file.
This includes desired XY, last sent u, commanded XY, ratio and angle. **Load u**
restores a matching file and refreshes the table; **Clear u** returns the context
to zero. Load/Clear are locked during execution. Saving does not turn Control off;
later corrections may refine memory, while the saved file keeps the snapshot.
The control details show speed as `measured (limit)` in mm/s within one fixed cell.

Precalculate all uses the current `p_d + u_last` snapshot. If learning changes
u, old angles cannot be reused: run falls back to fresh calculation until
precalculated again. For recording, train the path, restart it with Use learned
u, and optionally Precalculate all first. Keep Control on for refinement or off
for a fixed learned replay. This reduces repeated correction transients but
does not add continuous path interpolation or guarantee smooth motor motion.

## Execution

The original point plus any enabled learned u is sent first (precalculated if available). Its row wait
still applies. Then feedback uses the centroid of a complete detected group.
Each correction is solved freshly in a worker with the row's ratio, equilibrium
angle and motor zero; corrections do not reuse the original cached pose.
Only one solve/send is in flight. Camera and serial work stay under the existing
motor-command lock. **Allow beyond pins (extrapolated)** applies to sequence targets,
feedback corrections and replayed learned corrections alike. When checked, these
targets may extend beyond the four pins; when unchecked, the pin boundary is enforced.
All targets must remain inside the camera frame. The setting is captured when a run starts.

Observations must be no older than one second. Before sending, the observation
must still be valid, the group must not have moved by more than the arrival
tolerance, no tracking loss may have occurred during calculation, Control must
still be on and Stop must not have been pressed. Late solver results are
discarded. If optimization takes longer than the freshness window, corrections
may all be discarded; this is intentional protection, not a high-rate motor loop.

Missing/ambiguous tracking pauses corrections and resets transient history
without discarding the applied u. Turning
Control off stops further corrections; it does not undo the last motor pose.
Once all expected robots are within tolerance of the ORIGINAL desired point,
corrections pause. Next is manually available as soon as the original motor pose
is sent, whether or not arrival is confirmed. It can be pressed during the row
wait; advancement occurs after that wait. A queued Next also suppresses unsent
PID corrections. If robots leave tolerance, feedback can resume unless Next
was requested. Transient controller history resets per point; learned u is per row.

## Orientation and ratio

Robot orientation is NOT measured by the current centre/blob detector. There
is no orientation PID, inferred heading or measured shape-ratio feedback. The
sequence's ratio and equilibrium angle remain fixed. Ratio describes the
equilibrium shape; angle is its orientation, not a motor angle. With ratio 1,
the existing solver ignores angle. Group-centroid position control cannot
independently control each robot or guarantee a group orientation.

## Display and CSV

Main and pop-out views show desired XY, measured centroid, last applied u,
last commanded XY, proposed XY, ratio/angle, controller state, newly calculated
motor angles and LAST SENT motor angles separately. The status/hint area remains
two lines. Tracking and recording CSVs include the correction, commanded/proposed
XY, calculated/sent motor angles and send time (monotonic seconds). A calculated
angle is not proof that it was sent. Blank orientation is intentional. CSV
includes control_mode=incremental and control_alpha; control_kp remains as a
legacy-compatible alias for Alpha.

## Circular video export

Enable **Circle crop** before Start recording. **Fit workspace (4 pins)** uses
the minimum enclosing circle of the calibrated pins, rounded outward so all
four pins fit. Otherwise set **Radius px** in native camera pixels and **Place
crop**, then click either camera view for its center (Place crop unchecks Fit).
A thin blue circle previews the export boundary; this is NOT the tracking ROI.
The pop-out also has a Crop action. Disable Circle crop for full-frame export.

All three exported videos use the same fixed square of diameter 2*radius, with
outside-circle pixels black (blue for chroma-key overlays). Areas beyond the
source frame are padded, not stretched. Geometry is frozen and crop controls
locked during recording. CSV remains in native source pixels and calibrated
physical coordinates, unaffected by crop/zoom. Cropping is applied AFTER
overlays and never changes detection, calibration, desired points or control.

Files: `position_control.py` (PID/feedback), `command_sequences.py` (point loop),
`command_tab.py` (controls/display/CSV), `arrival_control.py` (fresh observations),
`settling_control.py` (speed estimation and continuous hold),
`learned_corrections.py` (context-scoped memory/files), `command_recording.py`
(native-coordinate overlays and circular recording crop).
