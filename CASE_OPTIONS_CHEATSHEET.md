# Case `PARAMS` Options Cheat Sheet

This is the central reference for options supported inside a case `PARAMS`
dictionary. Option names are case-sensitive.

## How case inheritance works

For a condition such as `cases/case_001/cond_006.py`, values are loaded in this
order:

1. `cases/case_001/case.py` supplies the base values.
2. `cases/case_001/cond_006.py` overrides only the values it defines.

A condition therefore does not need to repeat every required option.

## Usage summary

| Usage | Purpose | Main option groups |
|---|---|---|
| Usage 0 | Plot angles loaded from a sequence file | Core geometry, magnet properties, grid, `PLOT_TYPE` |
| Usage 1 | Save one field-free overview of the scheduled target trajectory | Targets, grid, workspace geometry |
| Usage 2 | Optimize targets and save angle sequences | Core options, targets, optimization |
| Usage 3 | Optimize targets and save static field plots | Usage 2 plus plot options |
| Usage 4 | Optimize, simulate dynamics, and save animation | All dynamics, payload, wall, solver, and animation options |

`Required` below means the merged base-case and condition dictionaries must
contain the key. A condition may inherit it from `case.py`.

## Core geometry, magnetic, and grid options

| Option | Default | Used by | Meaning |
|---|---:|---|---|
| `NUM_SOURCES` | Required | 1–4 | Number of external magnets; positive integer. |
| `RADIUS` | Required | 1–4 | Radius in metres of the circular external-magnet arrangement. |
| `SOURCE_MAGNETIZATION` | Required | 1–4 | Source magnet magnetization in A/m. |
| `ROBOT_MAGNETIZATION` | Required | 1–4 | Microrobot magnetization in A/m. |
| `L_SOURCE` | Required | 1–4 | Characteristic source-magnet length in metres. |
| `L_ROBOT` | Required | 1–4 | Characteristic robot length in metres; also sets robot volume and radius. |
| `GRID_MIN` | Required | 0, 1, 3, 4 | Minimum x/y plot and field coordinate in metres. Must be below `GRID_MAX`. |
| `GRID_MAX` | Required | 0, 1, 3, 4 | Maximum x/y plot and field coordinate in metres. |
| `RESOLUTION` | Required | 1, 3, 4 | Field-grid samples per axis; integer of at least 2. Higher values cost more time. |
| `INITIAL_ROBOT_POSITIONS` | Required | 2–4 | Array-like robot starting positions with shape `(number_of_robots, 2)`. |

## Target schedule and optimization

| Option | Default | Used by | Meaning |
|---|---:|---|---|
| `TARGET_SCHEDULE` | Required | 2–4 | Time-ordered equilibrium targets; formats are shown below. |
| `STABILITY_TRACE_MARGIN` | Scaled `1e-6` | 2–4 | Minimum trace margin used by stable-equilibrium optimization. |
| `STABILITY_DET_MARGIN` | Scaled `1e-12` | 2–4 | Minimum determinant margin used by stable-equilibrium optimization. |
| `OPTIMIZATION_FAILURE_MODE` | `"warn"` | 2–4 | `"warn"` continues with the best candidate; `"error"` raises immediately. |
| `TWO_EQUILIBRIUM_SOLVER` | `"stable"` | 2–4 | Solver for two targets: `"stable"`, `"plain"`, or `"center_repulsion"`. |

Supported `TARGET_SCHEDULE` entries:

```python
# One equilibrium; ratio must be positive and angle is in radians.
(start_time, target_position, eigenvalue_ratio, eigenvector_angle_rad)

# Two simultaneous equilibria.
(start_time, target_position_1, target_position_2)

# Four simultaneous equilibria.
(start_time, target_position_1, target_position_2,
 target_position_3, target_position_4)
```

Start times must be strictly increasing and every position must contain exactly
two coordinates.

## Static plot options

| Option | Default | Used by | Meaning |
|---|---:|---|---|
| `PLOT_TYPE` | `"force_info"` | 0, 3 | `"force_info"`, `"force_potential"`, or `"force_magnetic"` (also accepts `1`, `2`, `3`). |
| `PLOT_DRAW_TARGET_PATH` | `True` | 3 | Draws the full scheduled path when at least two schedule entries exist. |
| `PLOT_DISPLAY_SECONDS` | `1.5` | 3 | Seconds each `force_info` plot remains visible before advancing. |
| `TARGET_TRAJECTORY_TITLE` | `"Scheduled Target Trajectory"` | 1 | Title of the field-free target overview image. |
| `PLOT_FIELD_INSIDE_DISH` | `False` | 3, 4 | Fades the field outside the circular dish. Requires a positive `DISH_RADIUS`. |
| `DISH_CENTER` | `(0.0, 0.0)` | 1, 3, 4 | Dish centre `(x, y)` in metres. |
| `DISH_RADIUS` | None | 1, 3, 4 | Dish radius in metres. Required when field clipping is enabled. |
| `DISH_OUTSIDE_FADE_ALPHA` | `0.86` | 3, 4 | Outside-dish white overlay opacity from `0.0` to `1.0`. |
| `SHOW_EXTERNAL_MAGNET_MOMENT_VECTORS` | `False` | 3, 4 | Shows external magnet moment arrows. |
| `MAGNET_MOMENT_ARROW_LENGTH` | `0.035` | 3, 4 | Moment-arrow length in plot coordinates; must be positive. |
| `MAGNET_MOMENT_ARROW_COLOR` | `"#d1495b"` | 3, 4 | Any Matplotlib-compatible arrow colour. |

The Usage 3 target path is cyan and dashed, with waypoint dots, a green start
square, and an orange destination star.

## Robot and fluid dynamics

| Option | Default | Used by | Meaning |
|---|---:|---|---|
| `DENSITY_NDFEB` | Required for 4 | 4 | Robot material density in kg/m³; used to calculate robot mass. |
| `FLUID_VISCOSITY` | Required for 4 | 4 | Dynamic fluid viscosity in Pa·s. |
| `ALPHA` | Required for 4 | 4 | Dimensionless drag correction factor. |
| `CAPILLARY_SIN_C` | Required for 4 | 4 | Robot–robot capillary model coefficient. |
| `GAMMA` | Required for 4 | 4 | Surface tension in N/m. |
| `ROBOT_INTERACTION_SCALE` | `1.0` | 4 | Multiplies only the combined robot–robot magnetic and capillary force. `0.25` retains 25%; must be non-negative. |
| `USE_OVERDAMPED_DYNAMICS` | `False` | 4 | Uses terminal-velocity dynamics instead of acceleration dynamics. |
| `DYNAMICS_SPEEDUP` | `1.0` | 4 | Multiplier applied to overdamped terminal velocity. |
| `T_SPAN` | Required for 4 | 4 | Simulation `(start_time, end_time)` in seconds. |
| `T_EVAL_POINTS` | Required for 4 | 4 | Number of output time samples; integer of at least 2. |
| `SOLVER_PROGRESS_INTERVAL` | Required for 4 | 4 | Minimum seconds between integration-progress messages. |
| `SOLVER_METHOD` | `"RK45"` | 4 | SciPy `solve_ivp` integration method. |
| `SOLVER_RTOL` | `1e-5` | 4 | Relative integration tolerance. |
| `SOLVER_ATOL` | `1e-8` | 4 | Absolute integration tolerance. |
| `SOLVER_MAX_STEP` | Infinity | 4 | Maximum solver step in seconds; reduce it to prevent tunnelling through thin walls. |

While Usage 4 is solving dynamics, press `Ctrl+C` once to stop gracefully.
The samples reached so far are animated and saved to
`outputs/<case>_<condition>_partial.mp4`; a previously completed video is not
overwritten.

## Walls and constrained paths

| Option | Default | Used by | Meaning |
|---|---:|---|---|
| `WALL_SEGMENTS` | `[]` | 1, 3, 4 | Path boundaries in Usages 1 and 3 and simulated walls in Usage 4. Each wall is `(start, end)` for two-sided behavior or `(start, end, inward_normal)` for a one-sided wall. |
| `WALL_STIFFNESS` | `0.0` | 4 | Wall restoring-force stiffness. |
| `WALL_DAMPING` | `0.0` | 4 | Wall-contact damping. |
| `WALL_INTERACTION_RANGE` | `0.0` | 4 | Extra distance in metres over which wall forces act. |
| `WALL_RECOVERY_DEPTH` | `0.0` | 4 | Maximum crossed-wall recovery depth for one-sided walls. |

Names such as `PATH_POINTS`, `SPIRAL_POINTS`, and `CORRIDOR_HALF_WIDTHS` are
case-file helper variables, not automatically loaded `PARAMS` options. Use
them to build `TARGET_SCHEDULE` and `WALL_SEGMENTS`.

## Payload options

All options in this group are required by Usage 4 after base/condition merging.
Cases without a real payload currently use a near-zero `PAYLOAD_RADIUS` and
zero interaction coefficients.

| Option | Default | Used by | Meaning |
|---|---:|---|---|
| `PAYLOAD_RADIUS` | Required for 4 | 4 | Payload radius in metres. |
| `PAYLOAD_HEIGHT` | Required for 4 | 4 | Payload height in metres. |
| `PAYLOAD_DENSITY` | Required for 4 | 4 | Payload density in kg/m³. |
| `PAYLOAD_DRAG_FACTOR` | Required for 4 | 4 | Payload drag relative to the robot drag coefficient. |
| `CONTACT_STIFFNESS` | Required for 4 | 4 | Robot–payload contact stiffness. |
| `CONTACT_DAMPING` | Required for 4 | 4 | Robot–payload contact damping. |
| `PAYLOAD_CAPILLARY_GAIN` | Required for 4 | 4 | Robot–payload capillary attraction gain. |
| `PAYLOAD_CAPILLARY_RANGE` | Required for 4 | 4 | Capillary interaction range; cutoff is calculated as three times this value. |
| `PAYLOAD_INITIAL_POS` | Required for 4 | 4 | Initial payload `(x, y)` position in metres. |
| `PAYLOAD_INITIAL_VEL` | Required for 4 | 4 | Initial payload `(vx, vy)` velocity in m/s. |

## Animation and video options

| Option | Default | Used by | Meaning |
|---|---:|---|---|
| `ANIMATION_DRAW_CONTOUR` | `True` | 4 | Shows the field contour background. |
| `ANIMATION_DRAW_SOURCES` | `True` | 4 | Shows source magnets in the animation. |
| `ANIMATION_DRAW_ACTIVE_TARGET` | `True` | 4 | Shows the active target marker. |
| `ANIMATION_DRAW_STREAMLINES` | `False` | 4 | Shows field streamlines. |
| `ANIMATION_DRAW_QUIVER` | `False` | 4 | Shows field direction arrows. |
| `ANIMATION_DRAW_TARGET_TRAJECTORY` | `False` | 4 | Shows the route formed by scheduled target positions. |
| `ANIMATION_DRAW_TARGET_POINTS` | `False` | 4 | Shows orange dots at every scheduled primary target position. |
| `ANIMATION_DRAW_TRAJECTORIES` | `False` | 4 | Shows individual simulated robot trails. |
| `ANIMATION_ROBOT_MARKER_SIZE` | `55` | 4 | Matplotlib robot marker area; must be positive. |
| `ANIMATION_FIGURE_SIZE` | `(8, 8)` | 1, 4 | Figure size `(width, height)` in inches. With DPI, both output pixel dimensions must be even. |
| `ANIMATION_TITLE` | `"Microrobot Swarm Dynamics"` | 4 | Animation title. |
| `SAVE_VIDEO` | `True` | 4 | Saves an MP4 when true. |
| `VIDEO_DPI` | `160` | 4 | Video resolution multiplier; must be positive. |
| `VIDEO_FPS` | `30` | 4 | Encoded frames per second; must be positive. |
| `VIDEO_CRF` | `18` | 4 | H.264 quality from `0` to `51`; lower means higher quality and larger files. |

## Present in cases but not implemented

| Name | Status |
|---|---|
| `ROBOT_HEIGHT` | Currently present in `case_002/case.py`, but no loader, optimizer, plotter, or simulator reads it. `L_ROBOT` still determines robot volume and radius. |

## Copyable condition template

This compact template relies on values inherited from the sibling `case.py`:

```python
import numpy as np


PARAMS = {
    "TARGET_SCHEDULE": [
        (0.0, np.array([0.00, 0.00]), 1.0, np.deg2rad(0.0)),
        (10.0, np.array([0.05, 0.00]), 1.0, np.deg2rad(0.0)),
    ],
    "INITIAL_ROBOT_POSITIONS": np.array([
        [-0.01, 0.00],
        [0.01, 0.00],
    ]),

    # Usage 3
    "PLOT_TYPE": "force_info",
    "PLOT_DRAW_TARGET_PATH": True,
    "PLOT_FIELD_INSIDE_DISH": True,

    # Usage 4 timing and integration
    "T_SPAN": (0.0, 20.0),
    "T_EVAL_POINTS": 300,
    "SOLVER_PROGRESS_INTERVAL": 0.5,
    "SOLVER_METHOD": "RK45",
    "SOLVER_RTOL": 1e-5,
    "SOLVER_ATOL": 1e-8,

    # Usage 4 display/output
    "ANIMATION_DRAW_TARGET_TRAJECTORY": True,
    "ANIMATION_DRAW_TRAJECTORIES": False,
    "ANIMATION_DRAW_CONTOUR": True,
    "ANIMATION_DRAW_STREAMLINES": False,
    "ANIMATION_DRAW_QUIVER": False,
    "SAVE_VIDEO": True,
    "VIDEO_DPI": 160,
    "VIDEO_FPS": 30,
    "VIDEO_CRF": 18,

    # Usage 4 initial payload state (also required when payload is disabled)
    "PAYLOAD_INITIAL_POS": np.array([10.0, 10.0]),
    "PAYLOAD_INITIAL_VEL": np.array([0.0, 0.0]),
}
```

Base physical, dish, fluid, and payload constants should normally remain in
`case.py`; put only experiment-specific overrides in `cond_XXX.py`.
