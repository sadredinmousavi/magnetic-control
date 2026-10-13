# Commands position control

Control is optional and off by default. Enabling it requires confirmation. It
does not move anything until a sequence runs. The existing Run confirmation,
camera/controller checks, arrival hold, Next/Finish and Stop remain in effect.

## Idea

In calibrated robot XY (millimetres):

`error = desired - measured`

`u = Kp*error + Ki*integral(error) + Kd*filtered_derivative(error)`

`commanded = desired + bounded(u)`

The desired sequence point never changes. PositionPID has separate X/Y state
with shared gains, a vector correction limit, a slew-rate limit, derivative
filtering and conservative integral anti-windup. Defaults are P=0.2, I=D=0,
maximum correction 2 mm, slew 1 mm/s and minimum update period 1 s. These are
conservative starting settings, not experimentally tuned gains. Test on your
hardware carefully. Settings are locked during a run.

## Execution

The original point is sent first (precalculated if available). Its row wait
still applies. Then feedback uses the centroid of a complete detected group.
Each correction is solved freshly in a worker with the row's ratio, equilibrium
angle and motor zero; corrections do not reuse the original cached pose.
Only one solve/send is in flight. Camera and serial work stay under the existing
motor-command lock. Correction targets must remain inside the pins and camera,
even if the original sequence allowed extrapolation.

Observations must be no older than one second. Before sending, the observation
must still be valid, the group must not have moved by more than the arrival
tolerance, no tracking loss may have occurred during calculation, Control must
still be on and Stop must not have been pressed. Late solver results are
discarded. If optimization takes longer than the freshness window, corrections
may all be discarded; this is intentional protection, not a high-rate motor loop.

Missing/ambiguous tracking pauses corrections and resets PID history. Turning
Control off stops further corrections; it does not undo the last motor pose.
Once all expected robots are within tolerance of the ORIGINAL desired point,
corrections pause. Next is manually available as soon as the original motor pose
is sent, whether or not arrival is confirmed. It can be pressed during the row
wait; advancement occurs after that wait. A queued Next also suppresses unsent
PID corrections. If robots leave tolerance, feedback can resume unless Next
was requested. PID history resets per point.

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
angle is not proof that it was sent. Blank orientation is intentional.

Files: `position_control.py` (PID/feedback), `command_sequences.py` (point loop),
`command_tab.py` (controls/display/CSV), `arrival_control.py` (fresh observations).
