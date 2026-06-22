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

Current example case:

- `cases/case_payload_baseline.py`

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

Recommended naming pattern:

- `case_payload_baseline.py`
- `case_payload_heavy.py`
- `case_swarm_large_radius.py`
- `case_target_shifted.py`


## How Case Loading Works

Both `usage3.py` and `usage4.py` load a case by module name.

If you do not pass a case name, they use the default:

```text
case_payload_baseline
```


## PowerShell Usage

Run the default case:

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
python usage1.py case_001_cond_001 outputs/case_001_test_001.txt
python usage2.py case_001_cond_001
python usage3.py case_001_cond_001
python usage4.py case_001_cond_001
```

Example with another future case:

```powershell
python usage3.py case_payload_heavy
python usage4.py case_payload_heavy
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

1. Create a new file inside `cases/`.
2. Give it a clear name, for example:

```text
cases/case_payload_light.py
```

3. Copy the structure from `cases/case_payload_baseline.py`.
4. Modify the values in `PARAMS`.
5. Run one of the usage scripts with that case name:

```powershell
python usage3.py case_payload_light
python usage4.py case_payload_light
```


## Important Notes

- Pass the case name without the `.py` suffix.
- The case loader imports from `cases.<case_name>`.
- `usage3.py` needs only the control/field-related parameters.
- `usage4.py` needs the full parameter set, including payload and solver-related values.
