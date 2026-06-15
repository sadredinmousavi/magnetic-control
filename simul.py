import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm

# ============================================================
# Qualitative SPION Swarm Simulation
# Scenario:
#   1) dispersed SPIONs
#   2) chain formation under magnetic field
#   3) enhanced tumbling under rotating field + gradient
#   4) disaggregation
# ============================================================

np.random.seed(7)

# -----------------------------
# User inputs
# -----------------------------
N = 80                 # number of SPION particles
box = 1.0               # simulation domain size
dt = 0.04               # time step [s]
T = 12.0                # total time [s]

B0 = 20e-3              # magnetic field magnitude [T] = 20 mT
field_freq_hz = 5.0     # rotating magnetic field frequency [Hz]
gradB_mag = 1.0         # magnetic field gradient magnitude [T/m]
gradB_dir = np.array([1.0, 0.2])  # gradient direction

# Phase timing
t_chain = 3.0           # chain formation phase ends [s]
t_tumble = 8.0          # tumbling phase ends [s]
t_disagg = T            # disaggregation phase ends [s]

# Interaction parameters, qualitative
noise_strength = 0.002
chain_strength = 0.045
repulsion_strength = 0.0006
disagg_strength = 0.055

# Convert gradient direction to unit vector
gradB_dir = gradB_dir / np.linalg.norm(gradB_dir)

# Number of frames
frames = int(T / dt)

# -----------------------------
# Initial particle state
# -----------------------------
pos = np.random.uniform(0.1, 0.9, size=(N, 2))
pos[:, 1] += 0.05 * np.sin(10 * pos[:, 0])
vel = np.zeros_like(pos)

history = []
B_history = []
phase_history = []


# -----------------------------
# Helper functions
# -----------------------------
def magnetic_direction(t):
    """
    Unit direction of the magnetic field.

    Phase 1:
        Static magnetic field along +x.

    Phase 2 and 3:
        Rotating magnetic field in xy plane.
    """
    if t < t_chain:
        return np.array([1.0, 0.0])

    phi = 2 * np.pi * field_freq_hz * (t - t_chain)
    return np.array([np.cos(phi), np.sin(phi)])


def magnetic_field(t):
    """
    Magnetic field vector B(t) in Tesla.
    """
    return B0 * magnetic_direction(t)


def get_phase(t):
    if t < t_chain:
        return "Phase 1: chain formation"
    elif t < t_tumble:
        return "Phase 2: enhanced tumbling"
    else:
        return "Phase 3: disaggregation"


def apply_bounds(p):
    """
    Reflective-like clipping boundary.
    """
    return np.clip(p, 0.03, 0.97)


# -----------------------------
# Simulation loop with progress bar
# -----------------------------
print("Running SPION swarm simulation...")

for k in tqdm(range(frames), desc="Simulation progress", unit="frame"):
    t = k * dt

    b = magnetic_direction(t)
    B = magnetic_field(t)

    force = np.zeros_like(pos)

    # Brownian-like noise
    force += noise_strength * np.random.randn(N, 2)

    # Pairwise qualitative dipole-like interactions
    for i in range(N):
        rij = pos - pos[i]
        dist = np.linalg.norm(rij, axis=1) + 1e-9

        # only nearby particles interact
        mask = (dist > 0) & (dist < 0.10)

        if not np.any(mask):
            continue

        rhat = rij[mask] / dist[mask, None]
        cosang = rhat @ b

        # Attraction is strongest when particles are end-to-end
        # along the magnetic field direction.
        along_field = np.abs(cosang)
        attraction = chain_strength * (along_field ** 4) * np.exp(-dist[mask] / 0.045)

        # Short-range repulsion prevents complete collapse.
        side_repulsion = repulsion_strength / (dist[mask] ** 2)

        pair_force = (
            attraction[:, None] * np.sign(cosang)[:, None] * b[None, :]
            - side_repulsion[:, None] * rhat
        )

        # During disaggregation, apply perpendicular repulsion.
        if t > t_tumble:
            disagg_fraction = (t - t_tumble) / (t_disagg - t_tumble)
            perp = np.array([-b[1], b[0]])

            pair_force += (
                disagg_strength
                * disagg_fraction
                * np.exp(-dist[mask] / 0.08)[:, None]
                * np.sign(rhat @ perp)[:, None]
                * perp
            )

        force[i] += np.sum(pair_force, axis=0)

    # Gradient pulling during enhanced tumbling
    if t_chain <= t <= t_tumble:
        force += 0.018 * gradB_mag * gradB_dir

    # Extra spreading during disaggregation
    if t > t_tumble:
        center = np.mean(pos, axis=0)
        spread = pos - center
        force += 0.015 * spread / (np.linalg.norm(spread, axis=1)[:, None] + 1e-6)

    # Overdamped update
    vel = 0.80 * vel + force
    pos = pos + dt * vel
    pos = apply_bounds(pos)

    history.append(pos.copy())
    B_history.append(B.copy())
    phase_history.append(get_phase(t))

history = np.array(history)
B_history = np.array(B_history)

print("Simulation complete. Creating animation...")


# -----------------------------
# Animation
# -----------------------------
fig, ax = plt.subplots(figsize=(7, 7))

ax.set_xlim(0, box)
ax.set_ylim(0, box)
ax.set_aspect("equal")
ax.set_xlabel("x position")
ax.set_ylabel("y position")

scatter = ax.scatter([], [], s=16)

# Magnetic field arrow
B_arrow = ax.quiver(
    [0.15],
    [0.15],
    [1.0],
    [0.0],
    angles="xy",
    scale_units="xy",
    scale=7,
    width=0.008
)

# Gradient direction arrow
grad_arrow = ax.quiver(
    [0.15],
    [0.08],
    [gradB_dir[0]],
    [gradB_dir[1]],
    angles="xy",
    scale_units="xy",
    scale=7,
    width=0.008
)

phase_text = ax.text(
    0.04,
    0.96,
    "",
    transform=ax.transAxes,
    fontsize=10,
    va="top"
)

field_text = ax.text(
    0.60,
    0.96,
    "",
    transform=ax.transAxes,
    fontsize=10,
    va="top"
)

legend_text = ax.text(
    0.04,
    0.04,
    "Top arrow: B-field direction\nBottom arrow: gradient direction",
    transform=ax.transAxes,
    fontsize=9,
    va="bottom"
)


def update(frame):
    t = frame * dt
    p = history[frame]
    B = B_history[frame]
    b = B / np.linalg.norm(B)

    scatter.set_offsets(p)

    B_arrow.set_UVC(b[0], b[1])
    grad_arrow.set_UVC(gradB_dir[0], gradB_dir[1])

    phase_text.set_text(
        f"{phase_history[frame]}\n"
        f"t = {t:.2f} s"
    )

    field_text.set_text(
        f"|B| = {B0 * 1000:.1f} mT\n"
        f"f = {field_freq_hz:.2f} Hz\n"
        f"Bx = {B[0] * 1000:.2f} mT\n"
        f"By = {B[1] * 1000:.2f} mT\n"
        f"|∇B| = {gradB_mag:.2f} T/m\n"
        f"∇B dir = [{gradB_dir[0]:.2f}, {gradB_dir[1]:.2f}]"
    )

    ax.set_title("SPION Swarm: Chain Formation, Tumbling, and Disaggregation")

    return scatter, B_arrow, grad_arrow, phase_text, field_text, legend_text


anim = FuncAnimation(
    fig,
    update,
    frames=frames,
    interval=40,
    blit=False
)

# Save animation with progress percent
output_name = "spion_swarm_with_field_arrows.gif"

print("Saving animation...")

with tqdm(total=frames, desc="Animation saving progress", unit="frame") as pbar:
    class ProgressPillowWriter(PillowWriter):
        def grab_frame(self, **savefig_kwargs):
            super().grab_frame(**savefig_kwargs)
            pbar.update(1)

    writer = ProgressPillowWriter(fps=25)
    anim.save(output_name, writer=writer)

print(f"Done. Saved as: {output_name}")

plt.show()