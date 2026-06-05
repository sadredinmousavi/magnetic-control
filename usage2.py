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
    find_equilibrium_inputs,
    find_stable_equilibrium_inputs,
    calculate_potential_hessian
)

def main():
    # =========================================================================
    # 1. DEFINE PARAMETERS (from MATLAB script)
    # =========================================================================

    # --- System Geometry ---
    NUM_SOURCES = 8
    RADIUS = 0.25  # meters

    # --- Magnetic Properties ---
    MAGNETIZATION = 868e3  # A/m
    MAGNET_VOLUME = 0.02 * 0.02 * 0.02  # m^3

    # Calculate the scalar magnitude of the magnetic moment
    m_source_magnitude = MAGNETIZATION * MAGNET_VOLUME
    m_robot_magnitude = m_source_magnitude # Assuming robot has same magnetic properties

    # Calculate the Force Constant (C_F) based on Yousefi2021.pdf
    C_F = (3 * mu_0 / (4 * np.pi)) * m_source_magnitude * m_robot_magnitude

    # --- Observation Space ---
    GRID_MIN = -0.3  # meters
    GRID_MAX = 0.3   # meters
    RESOLUTION = 50  # Number of points per axis

    # =========================================================================
    # 2. SET UP SYSTEM CONFIGURATION & FIND EQUILIBRIUM
    # =========================================================================

    # Generate the positions of the 8 source electromagnets
    source_positions = generate_circular_source_positions(NUM_SOURCES, RADIUS)

    # --- Define Desired Equilibrium Point ---
    desired_pos = np.array([0.1, 0.05])  # An arbitrary point off-center
    
    print(f"Finding optimal control inputs to create an equilibrium at: {desired_pos}")
    
    # Calculate optimal control inputs using constrained optimization
    # control_inputs_u = find_equilibrium_inputs(desired_pos, source_positions, C_F)
    control_inputs_u = find_stable_equilibrium_inputs(desired_pos, source_positions, C_F, ratio=1)
    
    print("Optimized control inputs (u):")
    print(np.round(control_inputs_u, 3))

    # To calculate the B-field, we need the source moment *vectors*.
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

    Fx = np.zeros_like(X)
    Fy = np.zeros_like(Y)
    Bx = np.zeros_like(X)
    By = np.zeros_like(Y)
    U_pot = np.zeros_like(X)

    # Assuming the robot's magnetic moment is aligned with the z-axis for 
    # the Yousefi 2D approximation, but interacting with the 3D field.
    # If the robot is purely a z-axis dipole: m_robot_vec = [0, 0, m_robot_magnitude]
    # For a planar approximation matching the Abbott formulas, we project it:
    robot_moment_vec = np.array([0.0, 0.0, m_robot_magnitude]) 

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

            # Pad the 2D field to 3D to do the dot product (assuming planar z=0)
            B_3d = np.array([field_vec[0], field_vec[1], 0.0]) 
            # Potential Energy U = -B * m
            U_pot[i, j] = -np.dot(B_3d, robot_moment_vec)

    # # =========================================================================
    # # 4. VISUALIZE THE FIELDS
    # # =========================================================================

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    # fig.suptitle('Magnetic Fields (Optimized for Equilibrium)', fontsize=16)

    # # --- Plot 1: Force Field (Yousefi Model) ---
    # force_magnitude = np.sqrt(Fx**2 + Fy**2)
    # strm1 = ax1.streamplot(X, Y, Fx, Fy, color=np.log10(force_magnitude + 1e-12), linewidth=1, cmap='viridis')
    # ax1.set_title('Force Field (Yousefi Model)')
    # ax1.set_xlabel('x (m)')
    # ax1.set_ylabel('y (m)')
    # ax1.set_aspect('equal', adjustable='box')
    # ax1.plot(source_positions[:, 0], source_positions[:, 1], 'ro', markersize=8, label='Source Magnets')
    # # Mark the desired equilibrium point
    # ax1.plot(desired_pos[0], desired_pos[1], 'gX', markersize=12, label='Equilibrium Point')
    # ax1.legend()
    # fig.colorbar(strm1.lines, ax=ax1, label='log10(Force Magnitude [N])')

    # # --- Plot 2: Magnetic Field (Abbott Model) ---
    # field_magnitude = np.sqrt(Bx**2 + By**2)
    # strm2 = ax2.streamplot(X, Y, Bx, By, color=np.log10(field_magnitude + 1e-12), linewidth=1, cmap='inferno')
    # ax2.set_title('Magnetic Field (B) (Abbott Model)')
    # ax2.set_xlabel('x (m)')
    # ax2.set_ylabel('y (m)')
    # ax2.set_aspect('equal', adjustable='box')
    # ax2.plot(source_positions[:, 0], source_positions[:, 1], 'ro', markersize=8, label='Source Magnets')
    # ax2.plot(desired_pos[0], desired_pos[1], 'gX', markersize=12, label='Target Point')
    # ax2.legend()
    # fig.colorbar(strm2.lines, ax=ax2, label='log10(Field Magnitude [T])')

    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()

    # # =========================================================================
    # # 4. VISUALIZE THE FIELDS (UPDATED WITH QUIVER)
    # # =========================================================================

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    # fig.suptitle('Magnetic Fields (Optimized for Equilibrium)', fontsize=16)

    # # --- Plot 1: Force Field (Yousefi Model) ---
    # force_magnitude = np.sqrt(Fx**2 + Fy**2)
    # # Normalize vectors so arrows are visible everywhere
    # Fx_norm = np.divide(Fx, force_magnitude, out=np.zeros_like(Fx), where=force_magnitude!=0)
    # Fy_norm = np.divide(Fy, force_magnitude, out=np.zeros_like(Fy), where=force_magnitude!=0)
    
    # # Plot background colors for magnitude and arrows for direction
    # mesh1 = ax1.pcolormesh(X, Y, np.log10(force_magnitude + 1e-12), cmap='viridis', shading='auto', alpha=0.5)
    # # Use step slicing (e.g., [::2, ::2]) to prevent arrows from being too crowded
    # ax1.quiver(X[::2, ::2], Y[::2, ::2], Fx_norm[::2, ::2], Fy_norm[::2, ::2], 
    #            color='black', pivot='mid', scale=30)
    
    # ax1.set_title('Force Field Direction (Yousefi Model)')
    # ax1.set_xlabel('x (m)')
    # ax1.set_ylabel('y (m)')
    # ax1.set_aspect('equal', adjustable='box')
    # ax1.plot(source_positions[:, 0], source_positions[:, 1], 'ro', markersize=8, label='Source Magnets')
    # ax1.plot(desired_pos[0], desired_pos[1], 'gX', markersize=12, label='Equilibrium Point')
    # ax1.legend()
    # fig.colorbar(mesh1, ax=ax1, label='log10(Force Magnitude [N])')

    # # --- Plot 2: Magnetic Field (Abbott Model) ---
    # field_magnitude = np.sqrt(Bx**2 + By**2)
    # # Normalize vectors
    # Bx_norm = np.divide(Bx, field_magnitude, out=np.zeros_like(Bx), where=field_magnitude!=0)
    # By_norm = np.divide(By, field_magnitude, out=np.zeros_like(By), where=field_magnitude!=0)

    # mesh2 = ax2.pcolormesh(X, Y, np.log10(field_magnitude + 1e-12), cmap='inferno', shading='auto', alpha=0.5)
    # ax2.quiver(X[::2, ::2], Y[::2, ::2], Bx_norm[::2, ::2], By_norm[::2, ::2], 
    #            color='white', pivot='mid', scale=30)

    # ax2.set_title('Magnetic Field Direction (Abbott Model)')
    # ax2.set_xlabel('x (m)')
    # ax2.set_ylabel('y (m)')
    # ax2.set_aspect('equal', adjustable='box')
    # ax2.plot(source_positions[:, 0], source_positions[:, 1], 'ro', markersize=8, label='Source Magnets')
    # ax2.plot(desired_pos[0], desired_pos[1], 'gX', markersize=12, label='Target Point')
    # ax2.legend()
    # fig.colorbar(mesh2, ax=ax2, label='log10(Field Magnitude [T])')

    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()


    # =========================================================================
    # 4. VISUALIZE THE FIELDS
    # =========================================================================

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Magnetic Force and Potential Well', fontsize=16)

    # --- Plot 1: Force Field (Yousefi Model) ---
    force_magnitude = np.sqrt(Fx**2 + Fy**2)
    strm1 = ax1.streamplot(X, Y, Fx, Fy, color=np.log10(force_magnitude + 1e-10), linewidth=1, cmap='viridis')
    ax1.set_title('Force Field on Microrobot')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_aspect('equal', adjustable='box')
    ax1.plot(source_positions[:, 0], source_positions[:, 1], 'ro', markersize=8)
    ax1.plot(desired_pos[0], desired_pos[1], 'gX', markersize=12, label='Equilibrium Point')
    fig.colorbar(strm1.lines, ax=ax1, label='log10(Force [N])')

    # --- Plot 2: Magnetic Potential (U) ---
    # Using a contour plot to show the "well"
    contour = ax2.contourf(X, Y, U_pot, levels=50, cmap='coolwarm')
    ax2.set_title('Magnetic Potential Energy (U)')
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.set_aspect('equal', adjustable='box')
    ax2.plot(source_positions[:, 0], source_positions[:, 1], 'ro', markersize=8)
    fig.colorbar(contour, ax=ax2, label='Potential Energy (Joules)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


        # ---------------------------------------------------------
    # RESULTS & STABILITY ANALYSIS
    # ---------------------------------------------------------
    print("\n" + "="*45)
    print(" OPTIMIZATION RESULTS")
    print("="*45)
    
    # 1. Control Inputs (Formatted nicely to 3 decimal places)
    formatted_u_rad  = np.array2string(control_inputs_u, formatter={'float_kind': lambda x: f"{x:.3f}"})
    print("Optimized control inputs (u)  in radians:")
    print(f"{formatted_u_rad }\n")
    
    control_inputs_deg = np.degrees(control_inputs_u)
    formatted_u_deg = np.array2string(control_inputs_deg, formatter={'float_kind': lambda x: f"{x:.3f}"})
    print("Optimized control inputs (u) in degrees:")
    print(f"{formatted_u_deg}\n")
    
    # 2. Net Force and Angle (in degrees)
    net_force = calculate_total_force_from_sources(
        source_positions, control_inputs_u, desired_pos, m_source_magnitude, m_robot_magnitude
    )
    force_mag = np.linalg.norm(net_force)
    force_angle_deg = np.degrees(np.arctan2(net_force[1], net_force[0]))
    
    print("Net Force at equilibrium point:")
    print(f"  Fx        = {net_force[0]:.5e} N")
    print(f"  Fy        = {net_force[1]:.5e} N")
    print(f"  Magnitude = {force_mag:.5e} N")
    print(f"  Angle     = {force_angle_deg:.2f}°\n")
    
    # 3. Hessian / Stability Constraints
    # (Make sure C_F is defined in this scope, or replace it with your actual constant variable)
    H = calculate_potential_hessian(desired_pos, source_positions, C_F, control_inputs_u)
    trace_H = np.trace(H)
    det_H = np.linalg.det(H)
    eigenvalues, _ = np.linalg.eig(H)
    
    print("Stability Constraints (Potential Hessian):")
    print(f"  Trace       = {trace_H:.5e}")
    print(f"  Determinant = {det_H:.5e}")
    print(f"  Eigenvalues = [{eigenvalues[0]:.5e},  {eigenvalues[1]:.5e}]")
    print("="*45 + "\n")



if __name__ == '__main__':
    main()
