# Commands: force PD/PID control

Automatic control uses position feedback, filtered-velocity damping and an
optional slow integral. **Disturbance learning and the stationary-only mode are
removed**, including their settings and stored estimates. Control defaults off;
Ki defaults to 0.00015 micro-newtons/(mm*s), a slow integral starting value.
Set Ki to zero for PD only. Manual XY, the open-loop optimizer
and saved XY offsets are unchanged.

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

The allocator matches TOTAL force `A(p_measured) c` to `F_request` in cosine space.
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
