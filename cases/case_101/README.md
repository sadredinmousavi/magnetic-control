# Case 101

Conditions 001–008 are scaled versions of Case 100's production conditions.
No tests or calibration conditions are included. The modules derive their
configuration from Case 100 without modifying it.

Each condition uses one uniform factor no greater than one so all scheduled
targets and initial robot centers satisfy `r <= 0.05 m` about `(0,0)`.
Already-contained geometry is left at its original size. Including initial
robot centers also reduces condition 001's circle because its initial swarm
extends beyond the original 50 mm circle.

Channel walls, cargo planar dimensions, and associated spatial interaction
ranges follow the same scale. Timing, target orientation, formation ratios,
source magnet placement, robot size, material properties, and cargo height
keep their original settings. Scaling changes the dynamics; the radius check
does not guarantee that simulated robots or the full cargo footprint never
cross the 50 mm boundary.

Inspect `SCALE_FACTOR` in each module or `GEOMETRY_SCALE_FACTOR` in its loaded
parameters. Example:

```powershell
python usage1.py case_101.cond_007
python usage4.py case_101.cond_007
```

## Case 101-only feed-forward compensation

`compensator.py` fits a 2D thin-plate-spline RBF error field from the 69
Cartesian rows of `20261007_012328_dwell_points_compensation_xy.csv`.
The relevant four numeric columns are preserved in `assets/calibration_xy.csv`,
so this case does not depend on a Downloads file or a generated outputs file.
All model calculations use millimetres; target schedules remain in metres.

After scaling, each Case 101 condition automatically uses a numerically
inverted command when its predicted measured position matches the desired
position within 0.05 mm. Commands are constrained to the calibration-command
convex hull, which lies within the 50 mm saturation limit. Path sizes, timing,
formation ratios, walls, cargo geometry and initial robot positions are not
changed further by compensation.

Some existing desired positions are unreachable within the measured region.
Their original commands are retained, marked as **uncorrected**, and reported
with a warning. A failed candidate is never silently sent as compensation.
This is partial compensation, not a claim that every target is corrected.
Reducing desired paths or extending calibration would be needed to cover more
targets. Other cases never invoke this compensator.

`DESIRED_TARGET_SCHEDULE` preserves the pre-compensation route, while
`TARGET_SCHEDULE` contains the commands used by the optimizer. The per-point
diagnostics are available as `FEEDFORWARD_COMPENSATION_REPORT`. Generate all
eight reports with:

```powershell
python -m cases.case_101.compensator
```

The error model predicts systematic measurement error; the reported residuals
are model predictions, not independently verified experimental results.
