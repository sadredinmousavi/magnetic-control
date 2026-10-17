# Commands: continuous force PID and equilibrium compensation

GUI005 defaults to **Continuous PID**, initially PD because Ki=0. It combines
continuous force allocation during motion with a stationary disturbance observer.
Uncheck Continuous PID before starting a run to use the previous stationary-only
compensation. The old incremental XY/Alpha controller is not used. Manual mode
and saved XY offsets remain separate. Control itself still defaults off.

## Continuous force PID (default)

The magnetic force requested at the latest measured position is:

F_request = Kp * deadband(error) + integral_force - Kv * filtered_velocity - d_hat

The position deadband is continuous: remove the arrival-tolerance radius from
the error vector, or use zero inside that radius. Velocity damping remains
active inside arrival tolerance when the robot is moving above the stationary
threshold. It does not wait for settling or arrival before braking.
The allocation matches TOTAL force A(p_measured)c to F_request, not an extra
force added to the baseline. Baseline stiffness is still preserved at p_desired.

The stationary observer updates d_hat only after a fixed-command hold below
the configured 0.5 mm/s threshold. During motion d_hat is held; -F_magnetic is
never treated as a moving disturbance estimate. The averaged stationary position
and last sent magnet inputs are used. A held motor pose below the motor deadband
does not reset settling, allowing the stationary observer to collect samples.

GUI gain units and starting settings:
- Kp: 0.01 micro-newtons/mm
- Ki: 0 micro-newtons/(mm*s); positive values enable residual integral correction
- Kv: 0.01 micro-newtons*s/mm, damping from measured velocity (no setpoint derivative)
- Period: 0.25 s, a maximum request rate of 4 Hz, not a guaranteed send rate
- Velocity filter: 0.15 s
- Angle rate: 20 degrees/s, also bounded by Max angle step (10 degrees)
- Integral force maximum: 0.5 micro-newtons per component
- Motor deadband: 0.1 degrees, suppressing repeated tiny angle changes

Core calculations use SI. These are starting settings, NOT experimentally
tuned gains or a stability guarantee. Velocity uses actual timestamps and fresh
position differences, with first-order filtering. Repeated frames do not generate
extra PID updates. Lost tracking pauses control and requires velocity reacquisition;
the integral is frozen rather than catching up over the observation gap.

Integral updates are committed only for an accepted allocation, never for a
preview/discarded calculation. Conditional anti-windup blocks integration into
unachievable force components; it permits unwinding. A held pose uses its actual
model force for this test, not the force of the unsent candidate. With Ki>0, a
new disturbance estimate rebases integral bias to avoid a feedforward jump,
subject to the integral bound. Ki=0 disables integral accumulation completely.

Only one solve/transmission runs at a time. While busy, newer camera observations
replace older observations in the gate; there is no growing control queue. The
next update uses the latest available frame. A result older than min(1 s,
2*Period), tracking loss, Next becoming due, Stop or a mode change vetoes sending.
Angle-rate limits use elapsed time since the last completed transmission.

The row Wait now gates NEXT-point navigation only in continuous mode. PID starts
immediately after the initial pose transmission, even during that wait. A queued
Next waits for the row deadline while PID remains active; it does not disable
braking for the rest of the wait. Other modes retain their original wait behavior.

## Stationary-only cycle (Continuous PID unchecked)

1. Send the normal point pose, optionally using its saved manual XY offset.
2. Keep all eight commands fixed while one robot settles.
3. Require fresh tracking, a full speed window with at least three frames,
   and low speed/residual motion continuously for the hold interval.
4. Estimate d_hat = -A(p_measured) cos(theta_LAST_SENT), using the averaged
   settled position. Positions are metres for the magnetic model.
5. Allocate c_new so A(p_desired) c_new approximately equals -d_hat at the
   ORIGINAL desired point. Preserve the baseline potential Hessian and
   respect cosine bounds [0,1] and the maximum actual angle change.
6. Recheck freshness, motion, tracking loss, mode and Next/Stop before sending.
7. Reset settling after transmission. The next estimate uses the NEW angles.

Default speed threshold is **0.5 mm/s**. Window, Hold and Min wait are each
0.5 s. Maximum angle step is 10 degrees per stationary correction.
Control defaults off. Preview only defaults off.
Settings are configurable and locked during a run. Arrival tolerance (1 mm
by default) stops further correction once the settled position is close enough.
A slow frame, duplicate observation or tracking loss cannot complete the hold.
The monitor checks residual motion as well as regression velocity to reject
oscillations that return to their starting position.

The allocator uses a normalized quadratic force-error objective and small
movement/baseline penalties. All three independent entries of the baseline
potential Hessian are constrained and checked after solving; its eigenvalues
must be positive. An angle-limited update may cancel only part of the force.
A failed, non-improving or infeasible solve holds the last pose and reports
its reason. It is consumed once for that fixed-command stationary epoch,
rather than retried every camera frame. New motion, a send or a Control toggle
allows a new estimate. This optional mode has no error integrator or during-motion braking.

**Preview only** calculates without sending corrections (once per stationary
epoch in stationary mode; per fresh eligible sample in continuous mode).
Initial path and manual commands still move normally.

## Functions and ownership

- functions_main.py: estimate_equilibrium_disturbance and
  find_force_compensated_inputs contain the magnetic math and constrained solve;
  MagneticForcePID contains velocity filtering, force PID and anti-windup.
- control_workflow.py: calculate_equilibrium_force_control constructs the case
  model; calculate_force_pid_control allocates total force during motion.
  EquilibriumForceFeedback contains the reusable state machine for both modes.
- experimental/point_control.py: serializes calculation and transmission.
- experimental/command_tab.py: widgets, display and service connections only.

The old position_control.py remains a legacy reusable module for compatibility;
GUI005 does not import it or expose its PID/adaptive controls.

The reference Hessian is evaluated at the desired point from its nominal
optimizer inputs and retained across corrections. If the initial point used a
shifted XY command, the nominal optimizer runs once at the original desired
point to establish the reference with its requested ratio/orientation; that
reference is cached. A non-positive reference is rejected.

## Manual XY, saved u and navigation

Manual mode and mouse editing stay live during execution. A click requests an
active-point update through the existing worker; newer unsent clicks supersede
older ones. While stopped, clicking drafts commanded XY and Run selected sends.
Save u accepts the latest actually sent manual/initial XY correction in memory
and the table. Export points + u writes all accepted XY offsets to JSON;
Load/Clear restore/reset them. Drafts are not exported.

Automatic force compensation directly changes eight magnet inputs. It has no
unique equivalent XY offset, does not overwrite saved u, and leaves Commanded
XY/C marker blank or hidden after application. Save u explains this distinction
instead of storing an old XY value as the force correction. Force corrections
are relearned on each visit; the XY-u JSON does not store them.

Control on exits Manual and enables force compensation. Control off holds the
last transmitted pose. Entering Manual cancels unsent automatic work.
Next can queue one advance during calculation without requiring arrival;
the row minimum wait still applies. While idle it runs the following row
(First point at the end). There are no routine hardware confirmation dialogs.
Actual camera/controller, calibration and target-validity checks remain.

## Display, logging and interpretation

Main/pop-out fixed cells show desired/measured XY, speed (threshold), hold,
disturbance and requested cancelling force in model micro-newtons, predicted
force residual, and calculated/sent angles. Hover reveals clipped content.
The CSV stores forces in N: disturbance_n, requested_force_n,
predicted_force_n, force_residual_n, magnet_cosine_inputs,
stiffness_eigenvalues_n_m, control_calculation_time_s, force_compensated,
plus positions, settling data and calculated/sent angles.
Continuous-mode CSV fields additionally include velocity_xy_mm_s,
position_error_mm, pid_force_n, integral_force_n, held_disturbance_n,
observation_age_s, control_cycle_time_s and force gains in SI. Cycle time
includes calculation/transmission; observation age is software frame-receipt
age, not verified camera-exposure-to-physical-actuation latency.

The force scale uses nominal case magnetic properties. Micro-newtons are
model estimates, not independently calibrated measurements. The disturbance
includes magnetic-model error and contact reactions; transfer from measured
to desired position assumes locally similar disturbance. Positive magnetic
curvature does not prove total stability if unknown force gradients matter.

Automatic force compensation requires one tracked robot; it does not infer
individual forces from a swarm centroid. Multi-robot tracking, ordinary path
execution and Manual remain available. Transmission completion is not motor
position confirmation. Motor commands are sequential. The shared **Repeat**
checkbox in the main Commands page and pop-out Control group selects one send
when unchecked, or exactly 2/3 identical packets per motor when checked.
Default is checked with 3, matching the former hard-coded triple transmission.
There is 10 ms between copies of a motor packet: for eight motors this adds
0/80/160 ms at 1/2/3 sends, excluding serial and physical motor delay.
The setting also applies to Calibration axis moves through the shared commander.
It can change during a run; a transmission snapshots its count before sending,
and settling starts only after the complete transmission finishes. The optimizer
does not run again for duplicate packets. Other GUIs keep their legacy default.
Continuous force PID adds during-motion damping. Its attainable bandwidth is
still limited by these delays and physical actuator response; it is not a hard
real-time system or predictive controller. Verify response experimentally.

## Diagnostic data collection

Use **Start diagnostics** beside Pop-out camera in Commands, or in the pop-out
**Data** group. Start before running the experiment; press **Stop diagnostics**
after reproducing the problem. It also saves on application close. Files go to
`experimental/diagnostics/YYYY-MM-DD_HH-MM-SS_microseconds/` automatically:

- `frames.csv`: one row per detected robot per GUI-accepted camera frame, including
  empty/failed-detection frames. Desired/measured XY, error, filtered velocity,
  force estimates, PID output, stiffness, calculated/sent angles, ROI, states and timing.
- `records.jsonl`: full frame snapshots, independent calculation/send/hold/error
  events and half-second heartbeats (including camera outages).
- `session.json`: initial/final calibration, sequence, settings, hardware model,
  units, software versions, record counts and completeness/error status.

Send the **whole folder**, not just the CSV, for diagnosis. The JSONL includes
controller snapshots even where failed detection leaves CSV tracking fields empty.
Motor events distinguish a proposed calculation from completed serial writes;
they do not confirm actual motor rotation. Timestamps use a shared monotonic
clock; camera receipt time is not sensor exposure time. Controller telemetry is
the latest available snapshot and may lag the frame. This logs received/processed
Commands frames, not every sensor exposure. It does not change gains or issue
motor commands. Background writing avoids video encoding; bounded-queue overflow
or disk errors are explicitly reported as incomplete recordings. Videos remain
an independent recording option. No image data or connection credentials are saved.

## Circular recording export

Circle crop and Fit workspace use a fixed native-pixel circular crop for all
three exported videos. Set Radius px and Place crop for a manual centre.
Outside pixels are black (blue in chroma-key overlay), with out-of-frame areas
padded. Crop is applied after overlays and does not change tracking/control
coordinates. Settings freeze while recording; CSV remains in native camera
pixels and calibrated physical coordinates. Show commanded point adds the C
marker for XY/manual commands; no fictitious XY marker is drawn for a direct
force correction.
