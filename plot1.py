import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PARAMETERS
# ============================================================

mu0 = 4 * np.pi * 1e-7      # permeability of free space [T·m/A]

# Magnetization
M = 1000e3                   # [A/m]

# External permanent magnet
L_pm = 0.02                 # PM side length [m]
r_pm = 0.25                 # PM-to-workspace-center distance [m]

# Microrobot
L_robot = 0.0005            # robot side length [m]

# Distance range for plotting
d_min = 0.0005              # minimum center distance [m]
d_max = 0.08                # maximum center distance [m]
n_points = 2000

# ============================================================
# MAGNETIC MOMENTS
# ============================================================

# Cube magnetic moments
m_pm = M * L_pm**3
m_robot = M * L_robot**3

print("External PM moment:")
print(f"m_pm = {m_pm:.4e} A·m²")

print("\nMicrorobot moment:")
print(f"m_robot = {m_robot:.4e} A·m²")

# ============================================================
# EXTERNAL MAGNETIC FORCE
# ============================================================

# Simplified vertical-dipole force model
# External PM pushing one robot

F_ext = (
    (3 * mu0 / (4 * np.pi))
    * (m_pm * m_robot)
    / (r_pm**4)
)

print("\nExternal magnetic force on one robot:")
print(f"F_ext = {F_ext:.4e} N")
print(f"F_ext = {F_ext*1e9:.2f} nN")

# ============================================================
# INTER-ROBOT MAGNETIC REPULSION
# ============================================================

# Center-to-center distance array
d = np.linspace(d_min, d_max, n_points)

# Dipole-dipole repulsive force
F_rr = (
    (3 * mu0 / (4 * np.pi))
    * (m_robot**2)
    / (d**4)
)

# ============================================================
# EQUILIBRIUM DISTANCE
# ============================================================

# Solve:
# F_rr(d_eq) = F_ext

d_eq = (
    (3 * mu0 / (4 * np.pi))
    * (m_robot**2)
    / F_ext
)**0.25

print("\nEquilibrium distance:")
print(f"d_eq = {d_eq:.4e} m")
print(f"d_eq = {d_eq*1000:.2f} mm")

# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(8, 5))

# Inter-robot force curve
plt.loglog(
    d * 1000,
    F_rr,
    linewidth=2,
    label=r"$F_{rr}(d)$ inter-robot repulsion"
)

# Constant external force
plt.axhline(
    F_ext,
    linestyle="--",
    linewidth=2,
    label=fr"$F_{{ext}} = {F_ext:.2e}\ \mathrm{{N}}$"
)

# Equilibrium distance
plt.axvline(
    d_eq * 1000,
    linestyle=":",
    linewidth=2,
    label=fr"$d_{{eq}} = {d_eq*1000:.1f}\ \mathrm{{mm}}$"
)

# Equilibrium point
plt.scatter(
    [d_eq * 1000],
    [F_ext],
    s=80,
    zorder=5
)

# Labels
plt.xlabel("Center-to-center distance d [mm]")
plt.ylabel("Force [N]")

plt.title(
    "Equilibrium Distance:\n"
    "External Magnetic Attraction vs Inter-Robot Repulsion"
)

plt.grid(True, which="both", linestyle="--", alpha=0.5)

plt.legend()

plt.tight_layout()
plt.show()