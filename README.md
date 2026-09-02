# Magnetic Microrobot Simulation

This project contains two main entry scripts:

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

Both `usage3.py` and `usage4.py` load a case by module name.

There is no hard-coded default case. If no case is supplied, a file-selection
dialog opens so an unintended case is not run silently.


## PowerShell Usage

Select a case interactively:

```powershell
python usage3.py
python usage4.py
```


usage1 --> gives the angles and plot static
usage2 --> gives the target points and calculate and save into sequence
usage3 --> gives the target points and calculate and plot
usage4 --> gives the target points and calculate and simulate and make animation

Run a specific case:

```powershell
python usage1.py cases/case_001/cond_001.py outputs/case_001_test_001.txt
python usage2.py cases/case_001/cond_001.py
python usage3.py cases/case_001/cond_001.py
python usage4.py cases/case_001/cond_001.py
```

## What Each Script Does

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
```
pip install -r ./experimental/requirements.txt
```


```
python ./experimental/gui001.py
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