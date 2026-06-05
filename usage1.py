import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import mu_0

# Import the physics and utility functions
from functions import (
    generate_circular_source_positions,
    calculate_total_force_from_sources,
    calculate_total_field
)

from functions_2 import (
    build_actuation_matrix,
    find_equilibrium_inputs
)

def main():
    # =========================================================================
    # 1. DEFINE PARAMETERS (from MATLAB script)
    # =========================================================================

    # --- System Geometry ---
    NUM_SOURCES = 8
    RADIUS = 0.25  # meters

    # --- Magnetic Properties ---
    # Magnetization (M) and volume (V) are used to find the magnetic moment (m).
    # m = M * V
    MAGNETIZATION = 868e3  # A/m
    MAGNET_VOLUME = 0.02 * 0.02 * 0.02  # m^3

    # Calculate the scalar magnitude of the magnetic moment for a single source magnet and the robot
    m_source_magnitude = MAGNETIZATION * MAGNET_VOLUME
    m_robot_magnitude = m_source_magnitude # Assuming robot has same magnetic properties

    # --- Observation Space ---
    GRID_MIN = -0.3  # meters
    GRID_MAX = 0.3   # meters
    RESOLUTION = 50  # Number of points per axis

    # =========================================================================
    # 2. SET UP SYSTEM CONFIGURATION
    # =========================================================================

    # Generate the positions of the 8 source electromagnets
    source_positions = generate_circular_source_positions(NUM_SOURCES, RADIUS)

    # --- Define a Control Scenario ---
    # For this visualization, we'll set a static control input.
    # Let's assume all magnets are turned on to 'push' towards the center.
    # u = -1 means the force is directed from the point towards the magnet (attractive).
    # This corresponds to a radially inward magnetic moment vector.
    psi_deg = 45
    psi_rad = np.radians(psi_deg)
    u_value = np.cos(psi_rad)
    control_inputs_u = u_value * np.ones(NUM_SOURCES)

    # To calculate the B-field, we need the source moment *vectors*.
    # We'll align them radially based on the control inputs.
    # The moment vector m_vec = u * m_magnitude * (position / |position|)
    source_moment_vectors = np.zeros_like(source_positions)
    for i, pos in enumerate(source_positions):
        radial_unit_vector = pos / np.linalg.norm(pos)
        source_moment_vectors[i] = control_inputs_u[i] * m_source_magnitude * radial_unit_vector

    # =========================================================================
    # 3. CREATE GRID AND CALCULATE FIELDS
    # =========================================================================

    # Create a grid of points to sample the field
    x = np.linspace(GRID_MIN, GRID_MAX, RESOLUTION)
    y = np.linspace(GRID_MIN, GRID_MAX, RESOLUTION)
    X, Y = np.meshgrid(x, y)

    # Initialize arrays to store the vector field components
    Fx = np.zeros_like(X)
    Fy = np.zeros_like(Y)
    Bx = np.zeros_like(X)
    By = np.zeros_like(Y)

    # Iterate over each point in the grid
    for i in range(RESOLUTION):
        for j in range(RESOLUTION):
            point_pos = np.array([X[i, j], Y[i, j]])

            # Calculate the total force at this point using the Yousefi model
            force_vec = calculate_total_force_from_sources(
                source_positions,
                control_inputs_u,
                point_pos,
                m_source_magnitude,
                m_robot_magnitude
            )
            Fx[i, j] = force_vec[0]
            Fy[i, j] = force_vec[1]

            # Calculate the total magnetic field at this point using the Abbott model
            field_vec = calculate_total_field(
                source_positions,
                source_moment_vectors,
                point_pos
            )
            Bx[i, j] = field_vec[0]
            By[i, j] = field_vec[1]

    # =========================================================================
    # 4. VISUALIZE THE FIELDS
    # =========================================================================

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Magnetic Fields in the Workspace', fontsize=16)

    # --- Plot 1: Force Field (Yousefi Model) ---
    force_magnitude = np.sqrt(Fx**2 + Fy**2)
    # Use log scale for color to better visualize magnitude variations
    strm1 = ax1.streamplot(X, Y, Fx, Fy, color=np.log10(force_magnitude), linewidth=1, cmap='viridis')
    ax1.set_title('Force Field on Microrobot (Yousefi Model)')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_aspect('equal', adjustable='box')
    ax1.plot(source_positions[:, 0], source_positions[:, 1], 'ro', markersize=8, label='Source Magnets')
    ax1.legend()
    fig.colorbar(strm1.lines, ax=ax1, label='log10(Force Magnitude [N])')

    # --- Plot 2: Magnetic Field (Abbott Model) ---
    field_magnitude = np.sqrt(Bx**2 + By**2)
    strm2 = ax2.streamplot(X, Y, Bx, By, color=np.log10(field_magnitude), linewidth=1, cmap='inferno')
    ax2.set_title('Magnetic Field (B) (Abbott Model)')
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.set_aspect('equal', adjustable='box')
    ax2.plot(source_positions[:, 0], source_positions[:, 1], 'ro', markersize=8, label='Source Magnets')
    ax2.legend()
    fig.colorbar(strm2.lines, ax=ax2, label='log10(Field Magnitude [T])')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == '__main__':
    main()
