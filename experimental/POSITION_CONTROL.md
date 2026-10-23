# Commands: force PD/PID control

## GUI005: track cargo or robots, one at a time

**Control target** selects the only active detector in Commands and Trajectory
Track. **Robots** runs robot detection and shows robot markers/group geometry.
**Cargo center** runs cargo detection only: no robot detections, robot markers,
group geometry or complete-robot-count prerequisite. Changing the target stops
an active run and clears measurements/trails; results from the previous mode
cannot enter the new mode's feedback.
The larger camera window has a synchronized **Detection** selector. The
**Place cargo ROI** button is disabled in Robots mode and never switches the
detector implicitly. Changing modes also cancels pending ROI placement and clears
stale cargo status text.

For cargo, select **Cargo center**, press **Place cargo ROI** (or Place ROI),
and click to center the circle around the cargo. Adjust **ROI px** so it encloses
the cargo and as little unrelated material as possible. The detector searches
only inside this moving ROI, also respecting the workspace mask. It uses the
existing **Contrast** setting, regardless of cargo color; there is no red
threshold or color picking. Use **Cargo min area** to reject small dark fragments.
Local contrast outlines are filled to recover the solid silhouette for centroid
and shape fitting. The ROI follows the detected cargo and holds when it is lost.
With ambiguous objects, reduce/reposition the ROI around the intended cargo.

Circle/rectangle cargo geometry, calibrated cargo coordinates, arrival checking,
feedback and recording use this cargo measurement directly. A lost cargo pauses
corrections; robot positions are never substituted. Cargo CSV observations have
zero detected/expected robots, blank robot/group measurements and separate cargo
validity/position fields. The existing `RedCargoTracker` import remains an alias
for compatibility; the detector itself uses contrast, not red segmentation.

## GUI005 calibration using a pen: Manual axis lines

In Calibration, select **Method: Manual axis lines**. The existing **Symmetric
pairs** and **Center reference** robot calibration methods remain available,
and old setups continue to load. Changing methods resets the current axis fit;
save the existing setup first if you want to retain both calibrations as files.

1. Select the four workspace pins and enter the rectangle width/height as usual.
2. Press **Start / restart lines**. No robot, serial connection, motor preparation
   or tracking ROI is needed for the line calibration.
3. Put a pen along the physical X axis. Click a point toward **-X**, then another
   toward **+X** on the same straight edge or centerline of the pen.
4. Reposition the pen along the physical Y axis. Click **-Y**, then **+Y**.
5. Review the colored lines, signed endpoints, white intersection/origin and
   fitted positive-axis arrows. Save setup. **Undo line point** removes the most
   recent click so it can be replaced; restart clears all four.

Use well-separated points. They need not be equally spaced, any particular
distance apart, or on opposite sides of the origin. The order establishes
direction. Keep the camera and workspace fixed while repositioning the pen;
the view stays live so the second pen placement is visible. The enlarged camera
window accepts the same clicks and provides Axis lines / Undo line controls.

The intersection is calculated in the perspective-corrected metric plane,
not by taking pixel midpoints. Metric scale continues to come from the four-pin
rectangle. The existing orthogonal-axis fit preserves camera mirroring and
reports the angle between the two measured lines (90 degrees is ideal).
Near-parallel lines, coincident endpoints and an intersection outside the image
are rejected. A rejected click leaves previous clicks intact for another try.

The setup stores `axis_method: manual_lines`, the original four
`axis_line_points_px`, and canonical `axis_samples_px` for compatibility with
tracking, control and overlays. On load, it recomputes the fit from the original
clicks. Canonical positive references are synthetic 1 cm references; the pen
clicks are not treated as commanded robot travel. Camera/frame-size and metric
calibration changes invalidate the fit in the same way as robot-based axes.

Manual axes may be saved and loaded with `roi_circle_px: null`. Commands and
Trajectory Track can preview loaded XY points with completed axes before
selecting a hardware case, placing an ROI or connecting COM. Both the embedded
camera and larger camera window use the same projection; opening the larger
window refreshes calibration. Place an ROI later to enable tracking. There is
no fabricated active tracking ROI. Motor move/capture controls are disabled in the manual method. Line edits
are locked during control runs and motor preparation/movement. The separately
calibrated Helmholtz circle still defines its own center in the resulting frame.

### Camera recording before completing calibration

**Start recording** needs only a running camera with a fresh frame. It works
without COM, a hardware case, robot detection, an ROI, axes, a sequence or an
active trajectory. The larger camera window also offers Start recording and
Stop / save recording in its Data controls.

The recording still saves camera video, CSV and metadata for offline annotation.
Completed calibration and projected points are saved when available; tracking
and control measurements are included only when available. Frames without robot
measurements have an invalid-group CSV observation and blank unavailable values.
Metadata records whether tracking was available when recording started. Without
enough calibration to map XY, the raw camera image is recorded without invented
coordinates or point overlays. Physical motor execution retains its hardware,
serial connection, calibration and measurement checks.

## GUI005 trajectory default: Incremental XY PID (2026-10-21)

This section describes the new default in **Trajectory Track**. The force PID
described later belongs to the separate force-control path. The previous
`XY PID (current)` controller remains selectable. Usage scripts and other GUIs
do not select this new controller. Helmholtz radial compensation remains disabled;
its code and alignment settings are retained.

The new controller adjusts the existing optimizer command rather than adding
an absolute PID correction to the desired position. It does not assume a
one-to-one command-to-robot displacement. Internally, positions are metres,
velocity is metres/second, and time is seconds:

```text
e = waypoint - measured_position
delta_P = Kp * (e - previous_e)
delta_I = Ki * e * dt
delta_D = -Kd * (filtered_velocity - previous_filtered_velocity)
requested_command = previous_request + delta_P + delta_I + delta_D
```

`dt` uses fresh observation timestamps, capped at twice the control period.
Derivative is on measured motion, so changing the waypoint causes no derivative
kick. A run starts with the last sent motor angles and the measured start position
as the anchored solver's command origin. This is a local model coordinate, not a
claim that the physical robot sits at the nominal optimizer equilibrium. Command
and PID difference history carry across waypoints. A pause/tracking loss discards
pending changes and difference history, preserving the actual sent command;
resuming has no proportional/derivative catch-up kick. A new run seeds a new anchor.

Starting settings (experimental, not plant-identified tuning):

| Setting | Default | Meaning |
| --- | --- | --- |
| Kp | 0.25 | Dimensionless proportional gain |
| Ki | 0.02 /s | Continuous correction of remaining position error |
| Kd | 0.25 s | Damping from changes in filtered measured velocity |
| Command rate mm/s | 0.5 | Maximum norm of command displacement per elapsed second |
| Command offset max mm | 100 | Maximum radial command offset from the waypoint |
| Speed limit mm/s | 0.5 | Measured-speed braking threshold far from the waypoint |
| Slowdown mm | 3 | Reduce speed threshold inside this distance to waypoint |
| Brake gain s | 1.5 | Reverse-command amount proportional to measured velocity |
| Max brake mm | 2 | Brake proposal bound, before command-rate limiting |

The shared old `I max mm` input is disabled/unused in this mode. There is no
separate accumulating integral bucket: the bounded command stores the correction.
Rate/offset clipping discards excess increments, providing tracking anti-windup.
The original workspace validation and motor/optimizer feasibility restrictions
still apply; a 100 mm offset setting does not bypass them. If a waypoint moves
the offset bound beyond the current command, recovery toward it is rate-limited.

Command rate bounds the total XY vector, not each axis independently. Its budget
is limited by both observation elapsed time and actual time since the last write;
it cannot accumulate an unlimited burst during a pause. With the default period
of 0.25 s, ordinary updates are at most 0.125 mm. The maximum catch-up budget is
two periods. This controls command speed, not physical robot speed.

Measured-speed feedback is separate. The effective cap is
`speed_limit * clip(distance_to_waypoint / slowdown_distance, 0.1, 1)`.
Above that cap, waypoint P/I/D increments are suspended. The command moves
opposite measured velocity **relative to the sent command**, retaining its offset
from the measured position. Braking resumes normal PID below 80% of the cap;
brake commands also obey command-rate and offset limits. The controller holds
when position is inside arrival tolerance and speed is below both the effective
cap and the configured settling speed. Existing arrival hold-time checks remain.
Camera delay, mechanical lag and limited magnetic authority mean that neither a
hard physical speed ceiling nor zero overshoot is guaranteed.

Only a fully feasible, transmitted allocation advances the sent command. Partial
cosine-space allocation is rejected before transmission: its fraction is not
an exact XY displacement fraction. Failed/stale/preview calculations do not
advance command memory. Motor-deadband holds may retain a bounded pending request
so small corrections eventually cross the deadband; the displayed sent command
continues to identify the actual transmitted request. Motor write success is
not physical motor-position feedback.

Control details show desired, measured and sent commanded XY. The state includes
the proposed XY increment, effective speed cap and active rate/offset limit.
The legacy Integral display is blank in this mode. Tracking CSV adds
`command_increment_mm`, `command_offset_xy_mm`, `command_limits`,
`effective_speed_limit_mm_s`, `pid_p_increment_mm`, `pid_i_increment_mm`,
`pid_d_increment_mm` and `pending_increment_mm`. P/I/D fields describe the raw
normal-PID proposal; during braking they are diagnostic only and are not applied.
Recording metadata stores the selected controller and its settings.

Example from `20261021_135112_tracking`: with Y measured -7.050898 mm,
desired -10.526316 mm and sent command -4.274869 mm, error is -3.475418 mm.
If error and velocity are unchanged since the previous update, P and D
increments are zero. At Ki=0.02 /s and dt=0.25 s, the command advances by
-0.017377 mm to -4.292246 mm, subject to limits. It does not inherit the old
+7.122 mm integral. Starting a new run instead seeds the current measured
position and actual motor angles as a fresh local anchor.

Automated coverage includes send/hold/failure behavior, partial allocation,
rate/offset limits, slowdown/braking, pause/waypoint state and a synthetic plant
with 1.8x response, 0.5 s transport delay and 0.8 s time constant. This synthetic
test motivated a lower starting Kd than the earlier 1.5 s; it is not a validated
model of the physical experiment. The existing **Center** button remains its
explicit open-loop reset; use a trajectory target at (0, 0) for feedback-controlled
motion to the center.

## Existing force-control path

Automatic control uses position feedback, filtered-velocity damping and an
optional slow integral. **Disturbance learning and the stationary-only mode are
removed**, including their settings and stored estimates. Control defaults off;
Ki defaults to 0.00015 micro-newtons/(mm*s), a slow integral starting value.
Set Ki to zero for PD only. The shared open-loop optimizer and saved XY offset
handling are unchanged.

GUI005 also supports the deterministic Helmholtz radial-force compensation
described below. This is a supplied physical model, not disturbance learning.
When enabled it adjusts initial, manual and automatic GUI005 synthesis.

## Algorithm

Core calculations use metres, seconds and newtons:

```text
error = desired_position - measured_position
F_request = Kp * deadband(error) - Kv * filtered_velocity + integral_force
integral_candidate = clamp(integral_force + Ki * error * dt)
```

The continuous radial position deadband subtracts the arrival-tolerance radius
from the error magnitude, using zero inside that radius. Integral accumulates
only outside tolerance. Damping uses measured velocity rather than a derivative
of the setpoint, avoiding derivative kick on target changes.

With radial compensation off, the allocator matches TOTAL magnetic force
`A(p_measured) c` to `F_request` in cosine space. With it on, total force is
`A(p_measured) c + F_radial(p_measured)`; see the GUI005-only section below.
It does not add force to an already counted baseline force. The nominal optimizer
establishes the reference trap at the original desired point. All three independent
entries of its potential Hessian are preserved, retaining model stiffness, ratio
and orientation. Cosines stay in [0,1]; model angles are `acos(c)` in degrees,
without the hardware motor-zero offset.

The objective minimizes force mismatch with small movement/baseline penalties.
Actual angle-step bounds and positive restoring stiffness are validated. Invalid
or infeasible solves hold the last pose. Partial allocation is allowed and its
residual is displayed.

Integral commits only for an accepted allocation. It is bounded per component;
conditional anti-windup prevents accumulating into unachievable force and permits
unwinding. If the motor deadband holds a pose, anti-windup uses the force of the
ACTUALLY held angles, not the proposed angles. No estimated disturbance or integral
rebasing is applied, so becoming stationary cannot trigger a learned-force jump.

Velocity uses fresh observations and actual timestamps with first-order filtering.
After loss/staleness, two fresh samples are required again. Repeated timestamps
cannot produce duplicate updates. With Control on, bounded integral force carries
to each new point; velocity and integration timing reset, so a target change does
not introduce derivative kick or an integral catch-up step. Manual commands reset
integral. Control off/on or temporary loss retains bias but resets velocity history.

The next point's FIRST pose includes the inherited integral force. Its nominal
baseline is freshly synthesized or fetched from the validated precalculation cache,
then force is allocated at that target to include the carried bias while preserving
the new baseline Hessian and cosine bounds. Only the corrected pose is transmitted;
the nominal cache stays unchanged. This initial point transition uses the usual
initial-pose behavior, not the per-update PID angle-rate limit. Infeasible bias
allocation stops before transmission rather than sending a zero-bias fallback.
Later feedback allocates total P+I-D force at measured XY as before, so inherited
integral is counted once. Preview, Manual and Control off do not apply an inherited
force to the initial pose; Ki=0 clears integral memory.

Matching selected runs also inherit the last accepted integral, scoped to path,
calibration geometry/revision, camera identity, hardware model, robot count and
serial connection. New contexts or application restarts start fresh. **Reset PID
memory**, available in the main and camera views while stopped, clears memory
without moving motors. This is bounded force memory, not cumulative motor angles
or saved XY offsets. Integral remains adaptive and can unwind at the next point;
the bias needed at different physical positions may differ.

## Settings and use

Starting gains are not experimentally tuned or a stability guarantee:

- Kp: 0.015 micro-newtons/mm.
- Ki: 0.00015 micro-newtons/(mm*s), equivalent to a 100 s integral time at the
  default Kp. A constant 10 mm error adds 0.0015 micro-newtons per second before
  anti-windup or saturation. These are starting settings, not hardware-validated
  tuning; test a fixed point and inspect overshoot before running a path.
- Kv: 0.015 micro-newtons*s/mm.
- Period: 0.25 s, a maximum request rate of 4 Hz, not guaranteed send timing.
- Velocity filter: 0.15 s.
- Angle rate: 20 degrees/s; Max angle step also bounds each update (10 degrees).
- Integral maximum: 0.5 micro-newtons per component.
- Motor deadband: 0.1 degrees.
- Hold speed: 0.5 mm/s. A pose holds only when ALSO inside arrival tolerance
  (default 1 mm). Outside tolerance, low speed never pauses PID. Inside tolerance,
  motion above this threshold continues to receive damping.

Settings lock during a run. Manual and Control on/off stay live in both views.
Preview only calculates automatic corrections without sending; initial point
and manual commands still move. Automatic feedback requires one robot;
multi-robot tracking, ordinary execution and Manual remain available.

PID starts after the initial transmission, including during the row Wait. Wait
gates Next navigation, not feedback. Next can be queued without requiring arrival;
there is no automatic path progression. Arrival hold is only an arrival indicator,
not a stationary-force estimator or controller delay.

Before sending, recheck tracking, active point/mode, Stop/Next and observation age
(`min(1 s, 2*Period)`). Angle-rate limits use elapsed time since transmission end.

## Functions and ownership

- `functions_main.py`: `MagneticForcePID` filters velocity, calculates PID and
  anti-windup; `allocate_force_inputs` handles constrained magnetic allocation.
- `control_workflow.py`: `calculate_force_pid_control` builds the model and
  allocates only PID force; `ForcePIDFeedback` manages runtime state.
- `experimental/point_control.py`: serialized calculation and transmission.
- `experimental/command_tab.py`: widgets, display and service connections.

When the initial pose uses saved/manual XY u, nominal stiffness is calculated once
at the ORIGINAL desired point with its ratio/orientation and cached. A non-positive
reference is rejected. Automatic angle corrections have no unique equivalent XY
offset: they do not overwrite saved u or draw a fictitious commanded XY marker.
Save u keeps successfully sent manual/initial XY offsets in memory/table; Export
points + u exports these separately. Drafts and PID integral are not exported.
The legacy `position_control.py` is not used by GUI005.

## GUI005-only Helmholtz radial force

`gui005_radial_control.py` implements the supplied ideal-coil polynomial for a
fixed vertical robot moment at **z = +0.024 m relative to the coil-pair midpoint**:

```text
I_ref = 12 / 5.31 A
r = norm(p - coil_center)                 # metres
F_r = m_z * (I / I_ref) * (0.053654*r - 10.7146*r^3 - 1629.01*r^5)
F_radial = F_r * (p - coil_center) / r    # newtons; zero at r=0
F_total = A(p) c + F_radial(p)
```

Positive radial force points away from the coil axis. Signed current and signed
vertical moment both affect direction. For positive current and positive moment, the supplied
fit produces outward force over its supplied fit range, **0 <= r <= 0.040 m**,
measured about the configured coil center. At the user's request, GUI005 evaluates
the same polynomial and analytic stiffness beyond this range without a radius
cutoff or clamping. This is extrapolation; accuracy outside the supplied range is
unknown, and the polynomial can change force direction at larger radii. Finite
input/output validation, motor bounds, feasibility and total-stability checks
still apply.

For a finite expression at the axis, put `q = p - coil_center`, `s = q dot q`,
`g(s) = 0.053654 - 10.7146*s - 1629.01*s^2` and `k = m_z * I/I_ref`:

```text
F_radial = k * g(s) * q
H_radial = -k * (g(s)*identity + 2*g'(s)*outer(q,q))
g'(s) = -10.7146 - 3258.02*s
H_total = H_magnetic + H_radial
```

`H` is the potential Hessian (negative force Jacobian), in N/m. At the axis,
force is zero but `H_radial = -k*0.053654*identity` is generally nonzero.
Thus outward coil force reduces restoring stiffness even where force vanishes.

Initial point synthesis first obtains the ordinary nominal magnetic trap. A
GUI005-local constrained solve then balances total force to zero and preserves
the nominal trap's full Hessian as the **total** Hessian, retaining its strength,
eigenvalue ratio and orientation. Magnetic inputs remain in [0,1]. If that total
force/stiffness combination is infeasible, no fallback nominal pose is sent.
Initial equilibrium and carried integral force require a residual no greater than
max(1e-12 N, 1% of the requested force magnitude).

Force PID matches `A(p_measured)c = F_PID - F_radial(p_measured)` while retaining
the corrected reference total Hessian at the desired point. Coil compensation is
present once, including the first pose with carried integral. Requested force is
the desired **total** force; predicted/applied force and anti-windup residuals
include the coil force. The held-pose path uses total force from the actual angles.
Partial allocation during feedback still observes motor angle-step bounds.

The GUI005 XY optimizer controllers, including direction-weighted variants, use
differences between compensated baselines. An anchored pose still depends on the
actual initial motor pose and is not an assertion of exact zero force. Incremental
force allocation compensates at measured XY without an equilibrium solve or a
fixed-Hessian constraint. Total stiffness must be positive at the allocation
position for incremental force mode, and at the requested target for the other
compensated solves. These local checks do not prove physical closed-loop stability.

### Configuration and scope

Edit `experimental/gui005_radial_force.json`, then restart GUI005:

- `enabled`: currently **`false` at the user's request**, restoring ordinary GUI005
  calculations. The radial model, tests and coil-circle calibration remain in place
  for future work. Changing this setting requires restarting GUI005 and preparing
  fresh poses. Do not infer physical validation from the numerical model tests.
- `current_a`: signed current through each series coil; defaults to `12/5.31` A.
  These polynomial coefficients already correspond to that reference current.
- `mz_a_m2`: signed vertical magnetic moment in A*m^2. `null` uses
  `moment_sign * L_ROBOT^3 * ROBOT_MAGNETIZATION` from the selected hardware case.
  Set a number to override that value; for the illustrative cube use `0.000125`.
- `moment_sign`: `+1` or `-1`, applied only when `mz_a_m2` is `null`.
  The current GUI005 configuration restores **`+1` at the user's request after the
  inward-force trial**: positive configured current produces force away from the
  Helmholtz center over the supplied fit range, and magnetic compensation points
  inward. `-1` selects the inward-coil-force hypothesis instead. The physical
  sign remains subject to experimental validation. Restart GUI005 after changing it and
  prepare/recalculate poses for the new model; do not reuse old PID integral memory.
- `center_xy_m`: the coil-axis XY in GUI005's calibrated robot frame; defaults to
  `[0,0]`. This is a fallback when no valid coil-circle calibration is present;
  a calibrated Helmholtz circle overrides it in both GUI005 workspaces.

The GUI005 header displays whether compensation is on, the assumed height,
polynomial extrapolation and configured current. Both Control and Trajectory use matching
model settings, including calibration and precalculation. Model settings
enter preparation-cache, applied-pose and controller-memory signatures. Calculation
events include radial model parameters and radial force; force allocation events
also include magnetic and total forces and total stiffness eigenvalues.

### Aligning a Helmholtz axis separate from the workspace origin

In **2. Calibration**, select **Align Helmholtz circle (click / drag)**. Click
or drag on the camera image to move the blue circle and its center cross onto the
visible Helmholtz coil. Adjust **Circle radius px** (Enter, leaving the field, or
the spinner arrows applies it) to help align the ring. The pop-out camera has the
same **Coil** mode and **Coil r** field. The robot ROI remains independent.

After the four-pin calibration, the coil-center readout shows its XY in mm.
If robot axes are calibrated, the center is transformed into that robot frame;
otherwise it uses the rectangle-centered frame. Completing or changing the axes
recomputes the coil center from its camera location. For this axisymmetric model
only the center offset is needed; rotating XY axes does not change the force law.

The radius in pixels is a visual alignment guide, not a measured coil radius,
detection mask, crop, physical fit radius, or change to the force coefficients.
Radius is measured from this separate coil center. The supplied 40 mm fit range
is a reference for model accuracy; it imposes no runtime radius limit.

**Save setup** persists `helmholtz_circle_px` and the derived
`helmholtz_center_xy_m`. Loading recalculates the physical center using the saved
pins/axes; the pixel circle is authoritative. Alignment changes invalidate prepared
axis/sequence poses and change the model signature used by PID memory. Prepare
axis moves or precalculate the sequence again before using cached commands.
Alignment is locked during control runs, precalculation and calibration motor
operations. It never sends a motor command by itself.

**Clear coil alignment**, loading an older setup without a coil circle, changing
camera source, or a frame-size change restores the settings-file center until a
new valid alignment is available. Recalibrate the coil for each camera geometry.

Only `gui005.py` opts into this model through `PointCommander(radial_model=...)`.
A commander without that argument retains its prior calculations. Shared force
functions, `control_workflow.py`, usage scripts, case definitions, other GUIs and
their simulation/field plots do not acquire this force model.

The coefficients and the stated 40 mm fit range were supplied by the user for
Salehi's middle coil, ideal Helmholtz separation and 12 V across the series pair.
The reported approximately 0.2% fit agreement has not been independently verified
here. This model assumes a fixed vertical moment; it does not model induced or
tilted magnetization. Confirm the actual signed moment, current, coil center and
robot height before physical use. A different height or coil geometry needs new
coefficients; changing current alone scales this fit linearly. Hardware validation
is still required.

## Display, diagnostics and timing

Optional **Slow-motion assist (trial)** is off by default. Select it before a run
to increase proportional correction only after speed stays below 0.3 mm/s for
3 seconds, with error above max(3 mm, twice arrival tolerance). The proportional
multiplier ramps from 1 to 1.5 over another 3 seconds. Normal proportional gain
returns immediately when those conditions stop applying. Ki, Kv, accumulated
integral and all motor limits remain unchanged. Tracking loss, Control off,
manual mode and new points reset the dwell. Diagnostics log `slow_assist_active`,
`slow_assist_scale` and effective `force_kp_n_m`. This experimental setting does
not establish physical force direction or guarantee a settling time.

Both views use fixed cells for desired/measured XY, u, commanded XY, speed (hold
threshold), position error, integral, requested/allocated force, residual and
calculated/sent angles. Hover reveals clipped values. Force values are model
micro-newtons, not independently measured forces.

Use **Start diagnostics** beside Pop-out camera or in its Data group, reproduce
the problem, then **Stop diagnostics**. Sessions auto-save under
`experimental/diagnostics/YYYY-MM-DD_HH-MM-SS_microseconds/`, also on app close:

- `frames.csv`: per-robot rows for GUI-accepted frames, including failed/empty
  detections, desired/measured XY, speed/error, integral/PID force, allocation,
  stiffness, calculated/sent angles, ROI, states and timing.
- `records.jsonl`: full snapshots, independent calculation/send/hold/error events
  and half-second heartbeats, including camera outages.
- `session.json`: initial/final calibration, sequence, settings, model, units,
  versions, counts and completeness/error status.

Send the whole folder. CSV forces use N, positions mm. Measurement and force-source
timestamps identify source samples; telemetry may lag the current camera frame.
Monotonic receipt age is not sensor-exposure-to-actuation latency. It logs accepted
GUI frames, not every sensor exposure. Background writing avoids video encoding;
overflow/disk errors mark sessions incomplete. No images or connection credentials
are recorded. Old diagnostic files remain unchanged.

Motor commands are sequential. Repeat unchecked sends once; checked selects 2/3
identical packets per motor (default 3), without repeating optimization. Ten
milliseconds between copies adds 0/80/160 ms per eight motors, excluding serial
and physical response. Completed writes do not confirm physical rotation.
These delays limit bandwidth. Positive model stiffness does not prove physical
closed-loop stability. Hardware testing is required.

## Video export

Circle crop masks and enlarges the workspace in both live camera views and
limits robot detection to that circle. With four calibrated pins it automatically
encloses them with a 35-pixel margin, ignoring manual radius/center settings.
Without pins, Radius px/Place crop select a manual circle. Detection keeps its moving ROI
and real-image blur halo, with no artificial black edge fed to the detector.
Only the cropped image is converted for display; native tracking/CSV coordinates
and calibration remain unchanged. Turn Circle crop off for the full view.
Video recording uses the same fixed native-pixel circle for all three videos.
Outside pixels are black (blue for chroma-key overlay). Crop follows overlays
and does not change tracking coordinates. Crop settings freeze during recording;
CSV remains in native camera/physical coordinates.
