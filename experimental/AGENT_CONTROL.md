GUI005 can optionally expose measurements and its existing Commands controls to
a local agent. The original GUI files, motor protocol, calibration and workflow
are unchanged. No network service or extra dependency is installed.

Start `05-gui005-agent-control.bat` instead of the usual launcher when agent access
is wanted. Use the usual Connections controls to connect serial and camera.
This launcher opens the same GUI005, plus a local mailbox. Close any previous
GUI005 before connecting the same devices from the new instance.

From the project directory:

```powershell
.\.venv\Scripts\python.exe experimental/agent_control.py status
.\.venv\Scripts\python.exe experimental/agent_control.py snapshot
.\.venv\Scripts\python.exe experimental/agent_control.py show-camera
.\.venv\Scripts\python.exe experimental/agent_control.py load-setup --path experimental/setups/20261018_031759.json
.\.venv\Scripts\python.exe experimental/agent_control.py load-sequence --path experimental/xy_sequences/case_100_cond_013_test_extract_compensator.txt
.\.venv\Scripts\python.exe experimental/agent_control.py control --enabled true
.\.venv\Scripts\python.exe experimental/agent_control.py run-selected --point 1
.\.venv\Scripts\python.exe experimental/agent_control.py next
.\.venv\Scripts\python.exe experimental/agent_control.py stop
```

`run-all` starts the loaded sequence. Its usual per-point Next/Finish behavior
is preserved. `next` only advances an active run; it never starts an idle one.
`control` selects the existing force PID mode; it does not start a run itself.
Run acknowledgements report that execution started, not that motors arrived.
Inspect `status` for fresh positions, feedback and the per-point motor angles
and states. Commands work on the Tk thread, and the existing worker performs
calculation/feedback/transmission locally with its normal validation and camera
freshness checks. Agent communication is supervisory, not a guaranteed timing
source for the feedback loop.

Runtime data is stored under `outputs/gui005_control`, already ignored by Git.
Every GUI launch gets a new session. Requests expire after five seconds and are
claimed before execution so they are never automatically replayed. Stop takes
priority and invalidates older queued motion requests. Stop prevents further
commands; it does not remove magnetic force or undo the last motor pose.
The normal GUI Stop and window close remain available. Closing the agent
launcher marks its mailbox offline. Returning to the original launcher disables
agent access. The adapter does not connect hardware or start experiments by itself.

Agent-started runs open the existing Commands camera pop-out automatically, so
the operator can watch the camera, targets and tracked robots during control.
`show-camera` also opens it without moving anything. This is a window on the PC;
chat can retrieve snapshots, but does not display a continuous live video stream.

The reported camera delay is approximately one second. The optional launcher
uses that estimate by default (`gui005_agent.py --camera-delay 1`). Status clearly
distinguishes GUI receipt age from estimated physical image age: the camera
interface supplies no sensor capture timestamps. A freshly received frame can
therefore still show the scene from about a second earlier. The adapter waits
for a receipt after the previous send plus the estimated delay and 0.5 seconds
before accepting another supervised Run/Next command. This only gives the view
time to catch up; it does not prove that the robot has settled or compensate the
existing continuous PID loop. Start with Control off and observe each response.
Continuous feedback needs latency-aware tuning before use with this stream.
