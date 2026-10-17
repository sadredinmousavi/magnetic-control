# Commands: equilibrium force compensation

GUI005 Control now uses stationary force estimation and constrained cosine-space
allocation. Incremental XY/PID control, Alpha and adaptive-limit options are no
longer used by this GUI. Manual mode and saved XY offsets remain separate.

## Automatic cycle and defaults

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
allows a new estimate. There is no error integrator or during-motion braking.

**Preview only** calculates one result per stationary epoch without sending it.
Initial path and manual commands still move normally.

## Functions and ownership

- functions_main.py: estimate_equilibrium_disturbance and
  find_force_compensated_inputs contain the magnetic math and constrained solve.
- control_workflow.py: calculate_equilibrium_force_control constructs the case
  model. EquilibriumForceFeedback contains the reusable control state machine.
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

The force scale uses nominal case magnetic properties. Micro-newtons are
model estimates, not independently calibrated measurements. The disturbance
includes magnetic-model error and contact reactions; transfer from measured
to desired position assumes locally similar disturbance. Positive magnetic
curvature does not prove total stability if unknown force gradients matter.

Automatic force compensation requires one tracked robot; it does not infer
individual forces from a swarm centroid. Multi-robot tracking, ordinary path
execution and Manual remain available. Transmission completion is not motor
position confirmation. The sequential motor protocol, including explicit
sleeps, is unchanged. This compensates final offsets, not delayed braking.

## Circular recording export

Circle crop and Fit workspace use a fixed native-pixel circular crop for all
three exported videos. Set Radius px and Place crop for a manual centre.
Outside pixels are black (blue in chroma-key overlay), with out-of-frame areas
padded. Crop is applied after overlays and does not change tracking/control
coordinates. Settings freeze while recording; CSV remains in native camera
pixels and calibrated physical coordinates. Show commanded point adds the C
marker for XY/manual commands; no fictitious XY marker is drawn for a direct
force correction.
