# Magnetic Microrobot Simulation

This project contains five usage scripts:

- `usage0.py`
  Loads a previously saved magnet-angle sequence and renders its static field plots.
- `usage1.py`
  Saves one field-free overview of all scheduled target points and workspace geometry.
- `usage2.py`
  Computes optimized control inputs and saves the angle sequence.
- `usage3.py`
  Control-input workflow only. It computes the optimized control inputs for all scheduled targets and plots the results. It does not run dynamics simulation.
- `usage4.py`
  Full workflow. It computes control inputs for all scheduled targets, then runs the payload-aware dynamics simulation and animation.


## Case Files

Simulation and control parameters are defined in named case files under:

- `cases/`

Each case file is a Python module that defines one dictionary:

```python
PARAMS = {
    ...
}
```

Cases can also be organized as folders when several conditions share the same
physical/simulation parameters:

```text
cases/
  case_001/
    case.py       # shared parameters
    cond_001.py   # TARGET_SCHEDULE + INITIAL_ROBOT_POSITIONS
    cond_002.py
  case_002/
    case.py
    cond_001.py
  case_003/
    case.py
    cond_001.py
```

Run a folder condition with:

```powershell
python usage3.py cases/case_003/cond_001.py
python usage4.py cases/case_003/cond_002.py
```

Folder-style cases save with flat output names, for example:

```text
outputs/case_003_cond_002.txt
outputs/case_003_cond_002.mp4
outputs/case_003_cond_002/plot_001.png
```

## How Case Loading Works

All usage scripts load a case by module name.

There is no hard-coded default case. If no case is supplied, a file-selection
dialog opens so an unintended case is not run silently.


## PowerShell Usage

Select a case interactively:

```powershell
python usage3.py
python usage4.py
```


usage0 --> loads saved angles and plots their static fields
usage1 --> plots the complete scheduled target trajectory without a field
usage2 --> gives the target points, calculates, and saves the sequence in both
`outputs/` and `experimental/sequences/`
usage3 --> gives the target points and calculate and plot
usage4 --> gives the target points and calculate and simulate and make animation

Run a specific case:

```powershell
python usage0.py cases/case_001/cond_001.py outputs/case_001_test_001.txt
python usage1.py cases/case_001/cond_001.py
python usage2.py cases/case_001/cond_001.py
python usage3.py cases/case_001/cond_001.py
python usage4.py cases/case_001/cond_001.py
```

## What Each Script Does

### `usage0.py`

- loads magnet angles from a sequence created by Usage 2
- computes and displays a static field plot for each saved row

### `usage1.py`

- loads `TARGET_SCHEDULE` without running optimization
- plots every target track in schedule order
- retains source magnets, walls, and the dish outline
- saves one field-free image as `outputs/<case>_<condition>/target_trajectory.png`
- in the Windows launcher, choose `All cases and conditions` to skip the
  condition selector and save every plot without opening plot windows; batch
  files are named `outputs/target_trajectories/<case>_<condition>_target_trajectory.png`

### `usage3.py`

- loads the selected case
- computes stable control inputs for each target in `TARGET_SCHEDULE`
- prints optimization results for each target
- plots the optimized control inputs for all targets
- plots field results for each target

Use this when you only want control synthesis and visualization.


### `usage4.py`

- loads the selected case
- computes stable control inputs for each target in `TARGET_SCHEDULE`
- precomputes field data for each target
- builds the robot + payload initial state
- runs `solve_ivp`
- shows solver progress in the terminal
- animates the trajectories

Use this when you want the full dynamics simulation.


## Adding a New Case

1. Create a new folder inside `cases/`.
2. Add shared and condition files, for example:

```text
cases/case_006/case.py
cases/case_006/cond_001.py
```

3. Copy the structure from an existing `cases/case_*/case.py` and condition.
4. Modify the values in `PARAMS`.
5. Run one of the usage scripts with that case name:

```powershell
python usage3.py case_006/cond_001.py
python usage4.py case_006/cond_001.py
```


## Selecting Static Plot Types

Set `PLOT_TYPE` in a case or condition file:

```python
PARAMS = {
    "PLOT_TYPE": "force_info",
}
```

Supported values are `force_info`, `force_potential`, and `force_magnetic`.
All plotters consume the same precomputed field result and return a Matplotlib
figure. Changing the plot type does not repeat optimization or field sampling.

When usage scripts are imported, no dialog, case loading, optimization, or
plotting occurs. Callers can use `main(case_name=...)` programmatically.

Optimization failures emit a visible warning and continue with the best
candidate by default. To reject any optimizer-reported failure, set:

```python
PARAMS = {
    "OPTIMIZATION_FAILURE_MODE": "error",
}
```


## Animation Video Quality

`usage4.py` saves a square H.264 video using explicit quality settings. The
defaults produce a 1280 x 1280 video (`8 inches x 160 DPI`) with CRF 18. These
can be changed per case:

```python
PARAMS = {
    "ANIMATION_FIGURE_SIZE": (8, 8),
    "VIDEO_DPI": 160,
    "VIDEO_FPS": 30,
    "VIDEO_CRF": 18,
}
```

Lower `VIDEO_CRF` means higher quality and a larger file. Values from 16 to 23
are normally useful. Keep both values in `ANIMATION_FIGURE_SIZE` equal to
preserve a square video canvas.

`VIDEO_FPS` controls both the interactive preview and the saved video, so their
nominal playback speed is identical. If a computer cannot render a complex
preview in real time, reduce `VIDEO_FPS` (for example to 20); the saved file will
use the same rate.


## Important Notes

- Pass the case name without the `.py` suffix.
- The case loader imports from `cases.<case_name>`.
- `usage3.py` needs only the control/field-related parameters.
- `usage4.py` needs the full parameter set, including payload and solver-related values.























---
---

# Experimental

See [`CASE_OPTIONS_CHEATSHEET.md`](CASE_OPTIONS_CHEATSHEET.md) for every
implemented case `PARAMS` option, its default, and the usages that consume it.

Run these commands from the project root. Change `COM5` if the adapter uses a
different port.

### Setup and GUIs (PowerShell)

```powershell
python -m pip install -r .\experimental\requirements.txt
python .\experimental\gui001.py
python .\experimental\gui002.py
python .\experimental\gui003.py
python .\experimental\gui004.py
```

GUI004 is an offline microrobot detector. Select a recorded video, choose the
detection mode/color/minimum area, and process it without real-time playback
delays. It previews the latest annotated frame and saves an annotated MP4 and a
detection CSV under `outputs/offline_detection/<video_name>/`. It can also be
started by double-clicking `03-gui004.bat`. Its editable geometry defaults are
`h = 1 cm` and camera-to-dot-surface height `= 29 cm`; both values are recorded
in the detection CSV. The four green calibration dots are defined as a
`10 cm x 10 cm` rectangle by default; its width and height are independently
editable and are also recorded in the CSV. In Phase 1, use the play/pause button
or timeline to choose a clear raw frame, then click the four green dots in any
order. GUI004 orders and draws the corners, saves the local camera calibration
under `inputs/<video_name>_camera_calibration.json`, and masks later detection to that
four-point region. Each approximate calibration click searches a configurable
nearby radius and snaps to the nearest green-dot center. The calibration tab's
minimum-area setting can be lowered for very small dots. Phase 1 performs no
robot detection. In Phase 2, the separate click minimum-area setting and light
click-search filtering support very small robots without weakening full-video
detection filtering. The automatic detection cleanup size is also selectable;
use `1` for tiny robots and `3` or `5` to suppress progressively more noise.
After processing, Phase 4 replays the original video with green `+` robot marks,
a red `+` at the equal-weight mean of robot positions, and the workspace outline.
In Phase 2, enable **Circular detection area** to restrict automatic finding,
manual robot searches, and full-video detection to that circle. Use **Adjust
circle** to drag its center or edge, the diameter slider to resize, or **Fit to
workspace** for an initial placement. **Fit to workspace** returns directly to
robot-selection mode. When manually adjusting, uncheck **Adjust circle (blocks
selection)** before selecting robots.
The circle limits where color candidates may be found; it does not by itself
make every matching blob a robot. Phase 2 marks the accepted references as
`R1`, `R2`, and so on, and reports both the selected-robot count and the raw
color-candidate count. During processing, each robot identity is matched to the
nearest unclaimed candidate within the adjustable **Click/tracking radius**.
Because the four calibration dots and camera are fixed, candidates within the
adjustable **Calibration-dot exclusion** radius around P1-P4 are never promoted
to robots. They remain auditable in the CSV with
`exclusion_reason=calibration_marker`.
For experiments containing a painted red payload, tick **Detect red cargo** in
Phase 2. Cargo detection uses a separate red connected component and separate
minimum-area/tracking-radius controls, so enabling it does not change robot
selection. **Preview cargo** shows a cyan box and centroid around the measured
shape; both approximately circular and rectangular shapes are supported. The
largest red component initializes the cargo identity, after which frame-to-frame
position continuity is enforced. Cargo measurements are saved separately as
`<video_name>_cargo.csv`. Phase 4 can show or hide the cargo marker and its cyan
translation trail, and circular exports preserve those selected overlays.
Unmatched candidates are not drawn as robots or used for the red center, but
remain in the CSV with `accepted=0`; accepted rows include their `robot_id`.
With the circle disabled, detection covers the full frame, including outside
the green calibration lines. This detection region is separate from Phase 4's
export crop. Reprocess the video after changing the detection region; existing
CSV detections do not change. The red trail uses closely spaced 4-pixel dashes and
3-pixel gaps; this changes drawing density, not the measured sample count.
The center uses no temporal smoothing or detected-area weights. In Phase 4,
set **Expected robots** to the actual robot count: frames with a missing tracked
robot have no center marker and leave gaps in the center track. Raw candidate
positions and areas are preserved. A nearby false candidate can still replace
a missed robot, so inspect the numbered individual marks when interpreting the
trajectory. Reload existing detection CSVs to recompute their displayed
centers; previously saved annotated videos must be regenerated. In Phase 4,
robot markers can be switched between green `+` marks and hollow green
rectangles that leave the detected robots visible. Phase 4 also provides an
optional circular crop: drag inside the circle to move it, drag its edge or use
the diameter slider to resize it, or choose **Fit to workspace** to initialize
it around the calibration corners (limited to the frame size). The preview
dims the outside area. **Export cropped video** saves the entire original video
with the current overlay settings inside a square MP4 with black corners;
exports are silent and can be cancelled. Detection and calibration coordinates
remain in the original frame. Use the
Robot marks and Lines switches to show or hide those overlays, or load an
existing detection CSV for the selected video.
Phase 4 can also export transparent annotations. **Transparent AVI** uses
lossless RGBA PNG frames for editors that reject MOV; QuickTime Animation MOV
remains an alternate save-dialog format. **Export PNG sequence** is the most
portable fallback and includes `sequence_info.json` with the source frame rate.
For Camtasia, **Export blue-screen MP4** creates a standard high-quality H.264
video over pure blue (`#0000FF`). Put it above the original clip and apply
Camtasia's **Remove a Color** effect to the blue background.
Each option contains only the currently enabled workspace lines, robot/cargo
marks, labels, and translation trails. It retains the original dimensions,
frame rate, and frame count but contains no original image pixels or audio.
Place the overlay on a track directly above the untouched original at the same
start time; the original video therefore undergoes no quality-reducing re-encode.
The Red center track switch draws its path as a dashed line up to the current
frame; frames with no detections leave a break in the path.
Identify the robots separately by clicking them or by choosing their color.
Phase 3 processes
the complete video. A manual robot click searches the configurable nearby pixel
radius and snaps to the center of the nearest matching-color blob; if none is
found, the exact clicked position is retained and the GUI shows a warning.
`Learned from clicks` is the default: choose manual selection and click every
robot, and GUI004 builds an HSV profile from those samples for Find by color and
full-video processing. Learning also selects area `1` and cleanup size `1` so
tiny sampled robots are not discarded.

On Windows, double-click `04-usage.bat` in the project root. The launcher scans
`cases/case_*` and provides separate Usage, Case, and Condition menus. Use the
Up/Down arrow keys and Enter, or type a displayed list number/name and press
Enter. The selected module is passed directly to the usage, so no case-file
dialog opens. Usage 0 automatically uses the matching sequence from `outputs/`
or `experimental/sequences/`; run Usage 2 first if that sequence does not exist.

GUI001 and GUI002 start maximized, and their connection bars include controls
to connect or close the serial connection. GUI001 also includes a USB-webcam
viewer. Click `Webcam`; the GUI enumerates connected cameras by device name in a
background thread and lists them in the Camera dropdown. On Windows the label also
includes the USB VID/PID when available. Select the desired camera and press
Connect, or press Rescan after attaching new hardware. The viewer can overlay
OpenCV detections for a selected robot color,
`DICT_4X4_50` ArUco IDs, or both. `Min area` filters small color noise; increase
it when small false detections appear.

Close Windows Camera, Teams, browsers, and other camera applications before
opening or rescanning the GUI. Discovery performs warm-up reads and retries a
common MJPEG 640x480 mode for USB cameras whose first frames are not immediately
available. Exact Default-backend duplicates are hidden from the dropdown.

### Serial adapter tests (PowerShell)

```powershell
# TX voltage/scope test: disconnect the servo first
python .\experimental\test_serial_tx.py --port COM5 --baud 1000000 --seconds 3

# TX/RX loopback test: disconnect the servo and connect TX directly to RX
python .\experimental\test_serial_loopback.py --port COM5 --baud 1000000 --cycles 20

# Blink every connected Protocol 1.0 servo LED
python .\experimental\test_servo_led_broadcast.py --port COM5 --baud 1000000 --cycles 5 --delay 1

# Read servo IDs 1 through 8 without changing their settings
python .\experimental\test_u2d2_read.py --port COM5 --baud 1000000 --ids 1 2 3 4 5 6 7 8

# Try the MX-12W factory-default baud rate if no servo responds
python .\experimental\test_u2d2_read.py --port COM5 --baud 57600 --ids 1 2 3 4 5 6 7 8
```

Show all available options for a test:

```powershell
python .\experimental\test_serial_tx.py --help
python .\experimental\test_serial_loopback.py --help
python .\experimental\test_servo_led_broadcast.py --help
python .\experimental\test_u2d2_read.py --help
```

### Automated project tests (PowerShell)

```powershell
python -m unittest discover -s tests
```

### Raspberry Pi hardware scripts

These scripts currently use `/dev/ttyS0` or `/dev/serial0` in their source.

```bash
# Scan Dynamixel IDs 0-20 at common baud rates and both protocols
python ./experimental/scan_dynamixel.py

# Move servo ID 4 to approximately 90 degrees
python ./experimental/newmain.py

# Run the predefined multi-servo movement sequence
python ./experimental/main001.py

# Run the SDK helper example (LED and movement commands)
python ./experimental/main002.py
```






#
```
py -3.13 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests
```

### Installing requirements through a mirror

If the default PyPI server is slow or unavailable, pass a mirror URL to pip
with `--index-url` (or its short form, `-i`):

```powershell
python -m pip install -r requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

For the experimental tools, use the same option with their requirements file:

```powershell
python -m pip install -r .\experimental\requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

Replace the example URL with the HTTPS URL of your preferred PyPI-compatible
mirror.
