import numpy as np
import pandas as pd

# ============================================================
# CONSTANTS
# ============================================================

mu0 = 4 * np.pi * 1e-7      # permeability of free space [T·m/A]
M = 868e3                   # magnetization [A/m]

# ============================================================
# PARAMETER SETS
# ============================================================

# External PM side lengths [m]
L_pm_list = [0.02]

# PM-to-workspace-center distances [m]
r_pm_list = [0.15, 0.25]

# Microrobot side lengths [m]
L_robot_list = [0.0005, 0.00025]

# ============================================================
# STORAGE
# ============================================================

rows = []

# ============================================================
# LOOP OVER ALL PARAMETER COMBINATIONS
# ============================================================

for L_pm in L_pm_list:

    # External PM magnetic moment
    m_pm = M * L_pm**3

    for r_pm in r_pm_list:

        for L_robot in L_robot_list:

            # Microrobot magnetic moment
            m_robot = M * L_robot**3

            # ------------------------------------------------
            # External magnetic force
            # ------------------------------------------------

            F_ext = (
                (3 * mu0 / (4 * np.pi))
                * (m_pm * m_robot)
                / (r_pm**4)
            )

            # ------------------------------------------------
            # Equilibrium distance
            # Solve:
            # F_rr(d_eq) = F_ext
            # ------------------------------------------------

            d_eq = (
                (3 * mu0 / (4 * np.pi))
                * (m_robot**2)
                / F_ext
            )**0.25

            # ------------------------------------------------
            # Save results
            # ------------------------------------------------

            rows.append({
                "L_pm [m]": L_pm,
                "r_pm [m]": r_pm,
                "L_robot [m]": L_robot,
                "m_pm [A·m²]": m_pm,
                "m_robot [A·m²]": m_robot,
                "F_ext [N]": F_ext,
                "d_eq [m]": d_eq,
                "d_eq [mm]": d_eq * 1000
            })

# ============================================================
# CREATE TABLE
# ============================================================

df = pd.DataFrame(rows)

# Pretty formatting
pd.set_option("display.float_format", "{:.4e}".format)

print("\nEquilibrium Distance Table\n")
print(df)

# ============================================================
# OPTIONAL: SAVE TO CSV
# ============================================================

# df.to_csv("equilibrium_distance_table.csv", index=False)
# print("\nSaved to equilibrium_distance_table.csv")