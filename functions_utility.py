import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
from time import perf_counter
from pathlib import Path


from functions_main import *


class SolveIVPProgress:
    def __init__(self, t_span, min_interval=0.5):
        self.t_start = float(t_span[0])
        self.t_end = float(t_span[1])
        self.min_interval = min_interval
        self.last_print = perf_counter()
        self.last_percent = -1.0
        self.call_count = 0

    def update(self, t):
        self.call_count += 1

        span = self.t_end - self.t_start
        if span <= 0:
            percent = 100.0
        else:
            progress = (float(t) - self.t_start) / span
            percent = 100.0 * np.clip(progress, 0.0, 1.0)

        now = perf_counter()
        if (now - self.last_print) < self.min_interval and percent < 100.0:
            return

        if percent <= self.last_percent and percent < 100.0:
            return

        print(f"\rsolve_ivp progress: {percent:6.2f}% | t = {t:6.3f} s", end="", flush=True)
        self.last_print = now
        self.last_percent = percent

    def finish(self, status):
        print(
            f"\rsolve_ivp progress: 100.00% | status = {status} | rhs calls = {self.call_count}",
            flush=True
        )


def get_schedule_index(t, schedule):
    """
    Returns the active schedule index at time t.
    schedule entries must have start_time as their first element.
    """
    idx = 0

    for k, entry in enumerate(schedule):
        start_time = entry[0]
        if t >= start_time:
            idx = k
        else:
            break

    return idx

def compute_grid_fields(source_positions, control_inputs_u, source_moment_vectors, 
                        grid_min, grid_max, resolution, M_SOURCE_MAGNITUDE, M_ROBOT_MAGNITUDE):
    """Calculates the force field and potential energy across the observation grid."""
    x = np.linspace(grid_min, grid_max, resolution)
    y = np.linspace(grid_min, grid_max, resolution)
    X, Y = np.meshgrid(x, y)
    
    Fx = np.zeros_like(X)
    Fy = np.zeros_like(Y)
    Bx = np.zeros_like(X)
    By = np.zeros_like(Y)
    U_pot = np.zeros_like(X)
    
    # Robot moment aligned with the z-axis (planar approximation matching Abbott formulas)
    robot_moment_vec = np.array([0.0, 0.0, M_ROBOT_MAGNITUDE]) 
    
    for i in range(resolution):
        for j in range(resolution):
            point_pos = np.array([X[i, j], Y[i, j]])
            
            # Total force at this point
            force_vec = calculate_total_force_from_sources(
                source_positions, control_inputs_u, point_pos, 
                M_SOURCE_MAGNITUDE, M_ROBOT_MAGNITUDE
            )
            Fx[i, j] = force_vec[0]
            Fy[i, j] = force_vec[1]
            
            # Total magnetic field at this point
            field_vec = calculate_total_field(
                source_positions, source_moment_vectors, point_pos
            )
            Bx[i, j] = field_vec[0]
            By[i, j] = field_vec[1]
            
            # Potential Energy U = -B * m
            B_3d = np.array([field_vec[0], field_vec[1], 0.0]) 
            U_pot[i, j] = -np.dot(B_3d, robot_moment_vec)
    
    return X, Y, Fx, Fy, U_pot, Bx, By


def extract_optimization_info(
    control_inputs_u,
    net_force,
    H,
    eigenvalues,
    eigenvectors=None,
    desired_pos=None,
    equilibrium_positions=None,
    equilibrium_net_forces=None,
    center_repulsion_info=None,
    microrobot_positions=None
):
    """
    Extracts physical quantities from raw optimization results into a single dictionary.
    Extracts and computes physical quantities (angles, force magnitude) 
    from the raw optimization results.
    """
    # Calculate actual angles from u = cos(theta)
    angles_rad = np.arccos(control_inputs_u)
    angles_deg = np.degrees(angles_rad)
    
    # Calculate force metrics
    force_mag = np.linalg.norm(net_force)
    force_angle_deg = np.degrees(np.arctan2(net_force[1], net_force[0]))
    
    # Calculate Hessian metrics
    trace_H = np.trace(H)
    det_H = np.linalg.det(H)
    
    if microrobot_positions is not None:
        microrobot_positions = np.array(microrobot_positions)

    if equilibrium_positions is None and desired_pos is not None:
        equilibrium_positions = [desired_pos]

    if equilibrium_net_forces is None:
        equilibrium_net_forces = [net_force]
    
    return {
        'u_values': control_inputs_u,
        'angles_rad': angles_rad,
        'angles_deg': angles_deg,
        'net_force': net_force,
        'force_mag': force_mag,
        'force_angle_deg': force_angle_deg,
        'H_trace': trace_H,
        'H_det': det_H,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'desired_pos': desired_pos,
        'equilibrium_positions': equilibrium_positions,
        'equilibrium_net_forces': equilibrium_net_forces,
        'center_repulsion_info': center_repulsion_info,
        'microrobot_positions': microrobot_positions
    }


def print_optimization_results(opt_info):
    """Prints results using the bundled opt_info dictionary."""
    print("\n" + "="*45)
    print(" OPTIMIZATION RESULTS")
    print("="*45)
    
    formatted_u = np.array2string(opt_info['u_values'], formatter={'float_kind': lambda x: f"{x:.3f}"})
    print(f"Optimized control inputs (u = cos(theta)):\n{formatted_u}\n")
    
    formatted_angles = np.array2string(opt_info['angles_deg'], formatter={'float_kind': lambda x: f"{x:.3f}"})
    print(f"Actual Magnet Angles (theta) in degrees:\n{formatted_angles}\n")
    
    eq_positions = opt_info.get('equilibrium_positions') or [opt_info.get('desired_pos')]
    eq_forces = opt_info.get('equilibrium_net_forces') or [opt_info['net_force']]

    for idx, (eq_pos, nf) in enumerate(zip(eq_positions, eq_forces), start=1):
        force_mag = np.linalg.norm(nf)
        force_angle_deg = np.degrees(np.arctan2(nf[1], nf[0]))
        label = f" {idx}" if len(eq_positions) > 1 else ""
        print(f"Equilibrium Point{label}:")
        if eq_pos is not None:
            print(f"  Position  = ({eq_pos[0]:.5f}, {eq_pos[1]:.5f}) m")
        print(f"  Fx        = {nf[0]:.5e} N")
        print(f"  Fy        = {nf[1]:.5e} N")
        print(f"  Magnitude = {force_mag:.5e} N")
        print(f"  Angle     = {force_angle_deg:.2f} deg\n")

    center_info = opt_info.get('center_repulsion_info')
    if center_info is not None:
        center_pos = center_info['position']
        center_force = center_info['force']
        print("Center Detraction Point:")
        print(f"  Position            = ({center_pos[0]:.5f}, {center_pos[1]:.5f}) m")
        print(f"  Force Magnitude     = {np.linalg.norm(center_force):.5e} N")
        if 'trace' in center_info and 'determinant' in center_info:
            print(f"  Hessian Trace       = {center_info['trace']:.5e}")
            print(f"  Hessian Determinant = {center_info['determinant']:.5e}")
        if 'eigenvalues' in center_info:
            center_evals = center_info['eigenvalues']
            print(f"  Hessian Eigenvalues = [{center_evals[0]:.5e},  {center_evals[1]:.5e}]")
        print(f"  Line Curvature      = {center_info['line_curvature']:.5e}")
        print(f"  Line Repulsion      = {-center_info['line_curvature']:.5e}\n")

    print("Stability Constraints (Potential Hessian):")
    print(f"  Trace       = {opt_info['H_trace']:.5e}")
    print(f"  Determinant = {opt_info['H_det']:.5e}")
    evals = opt_info['eigenvalues']
    print(f"  Eigenvalues = [{evals[0]:.5e},  {evals[1]:.5e}]")
    print("="*45 + "\n")
    return

    nf = opt_info['net_force']
    print("Net Force at equilibrium point:")
    print(f"  Fx        = {nf[0]:.5e} N")
    print(f"  Fy        = {nf[1]:.5e} N")
    print(f"  Magnitude = {opt_info['force_mag']:.5e} N")
    print(f"  Angle     = {opt_info['force_angle_deg']:.2f}°\n")
    
    print("Stability Constraints (Potential Hessian):")
    print(f"  Trace       = {opt_info['H_trace']:.5e}")
    print(f"  Determinant = {opt_info['H_det']:.5e}")
    evals = opt_info['eigenvalues']
    print(f"  Eigenvalues = [{evals[0]:.5e},  {evals[1]:.5e}]")
    print("="*45 + "\n")


def get_target_at_time(t, target_schedule):
    """
    Returns active desired point at time t.
    target_schedule: list of (start_time, target_pos, eig_ratio, eigvec_angle_rad)
    """
    active_idx = get_schedule_index(t, target_schedule)
    return target_schedule[active_idx][1]


def get_control_at_time(t, target_controls):
    """
    Returns active target and control input at time t.
    target_controls:
        list of (start_time, target_pos, eig_ratio, eigvec_angle_rad, control_inputs_u)
    """
    active_idx = get_schedule_index(t, target_controls)
    return target_controls[active_idx][1], target_controls[active_idx][4]

def plot_mode_1(X, Y, Fx, Fy, source_positions,  opt_info, 
                draw_contour=True, draw_desired_point=True,
                plot_microrobots=True, plot_trajectories=False,
                block=True, display_seconds=None, reuse_window=False, **kwargs):
    """Mode 1: Info panel on left, force field (streamlines & contours) on right."""
    if reuse_window:
        fig = plt.figure("Plot Mode 1", figsize=(14, 7))
        fig.clf()
    else:
        fig = plt.figure(figsize=(14, 7))
    
    info_ax = fig.add_axes([0.05, 0.1, 0.25, 0.8])
    plot_ax = fig.add_axes([0.35, 0.1, 0.6, 0.8])
    
    F_mag = np.sqrt(Fx**2 + Fy**2)
    desired_pos = opt_info['desired_pos']
    equilibrium_positions = opt_info.get('equilibrium_positions') or [desired_pos]
    
    if draw_contour:
        finite_vals = F_mag[np.isfinite(F_mag)]
        finite_vals = finite_vals[finite_vals > 0]

        custom_levels = None
        F_mag_plot = np.array(F_mag, copy=True)

        if finite_vals.size > 0:
            min_val = max(np.min(finite_vals), 1e-12)
            max_val = np.percentile(finite_vals, 55)

            if not np.isfinite(max_val) or max_val <= min_val:
                max_val = np.max(finite_vals)

            if np.isfinite(max_val) and max_val > min_val:
                custom_levels = np.logspace(np.log10(min_val), np.log10(max_val), num=20)
                F_mag_plot = np.where(np.isfinite(F_mag_plot), F_mag_plot, max_val)
                F_mag_plot = np.clip(F_mag_plot, min_val, max_val)

        if custom_levels is not None:
            contours = plot_ax.contourf(
                X,
                Y,
                F_mag_plot,
                levels=custom_levels,
                cmap='viridis',
                extend='both',
                alpha=0.6
            )
            plot_ax.clabel(contours, inline=True, fontsize=8, fmt='%.1e')
            fig.colorbar(contours, ax=plot_ax, label='Force (N)')
    
    # Streamlines
    plot_ax.streamplot(X, Y, Fx, Fy, color='black', linewidth=1.2, density=1.5, arrowstyle='->')
    
    plot_ax.plot(source_positions[:, 0], source_positions[:, 1], 'r.', markersize=15, label='Magnets')
    
    if draw_desired_point:
        for idx, eq_pos in enumerate(equilibrium_positions):
            label = 'Equilibrium Point' if idx == 0 else None
            plot_ax.plot(eq_pos[0], eq_pos[1], 'rx', markersize=15, markeredgewidth=2, label=label)
    
    if plot_microrobots and opt_info.get('microrobot_positions') is not None:
        # bots = opt_info['microrobot_positions']
        # # If bots is a 1D array of a single coordinate [x, y], reshape it
        # if bots.ndim == 1:
        #     bots = bots.reshape(1, 2)
        # plot_ax.plot(bots[:, 0], bots[:, 1], 'ks', markersize=8, label='Microrobots')
        # Assuming your axes are in METERS. If your axes are in mm, use 0.5 instead.
        robot_length = 0.005  # 0.5 mm in meters
        
        for pos in opt_info['microrobot_positions']:
            # Calculate the bottom-left corner of the square so the robot is centered on 'pos'
            bottom_left_x = pos[0] - (robot_length / 2)
            bottom_left_y = pos[1] - (robot_length / 2)
            
            # Create a rectangle patch
            rect = patches.Rectangle(
                (bottom_left_x, bottom_left_y), 
                robot_length,  # width
                robot_length,  # height
                linewidth=1, 
                edgecolor='blue', 
                facecolor='blue',
                zorder=5 # Ensures they are drawn on top of streamlines
            )
            # Add the rectangle to the plot
            plot_ax.add_patch(rect)
    
    if plot_trajectories and opt_info and 'trajectories' in opt_info:
        trajectories = opt_info['trajectories']
        num_robots = len(opt_info['microrobot_positions'])
        
        for i in range(num_robots):
            # Extract X and Y history for this specific robot
            x_history = trajectories[i*4, :]
            y_history = trajectories[i*4+1, :]
            
            # Plot the path as a dashed line
            plot_ax.plot(x_history, y_history, color='orange', linestyle='--', linewidth=1.5, zorder=4)
    
    plot_ax.set_title('Force Field', fontsize=14, fontweight='bold')
    plot_ax.set_xlabel('x [m]')
    plot_ax.set_ylabel('y [m]')
    plot_ax.set_xlim([np.min(X), np.max(X)])
    plot_ax.set_ylim([np.min(Y), np.max(Y)])
    plot_ax.set_aspect('equal', adjustable='box')
    plot_ax.grid(True)
    
    # Information Panel Setup
    info_ax.axis('off')
    psai_text = "Magnet Angles:\n"
    for i, angle in enumerate(opt_info['angles_rad']):
        psai_text += f"Magnet {i+1}: {angle:.3f} rad ({opt_info['angles_deg'][i]:.1f}°)\n"
        
    eq_text = "\nEquilibrium Positions:\n"
    if equilibrium_positions:
        for idx, eq_pos in enumerate(equilibrium_positions, start=1):
            label = f"{idx}: " if len(equilibrium_positions) > 1 else ""
            eq_text += f"{label}({eq_pos[0]:.3f}, {eq_pos[1]:.3f})\n"
    else:
        eq_text = ""
    
    stability_text = "\nStability Analysis:\n"
    evals = opt_info['eigenvalues']
    evecs = opt_info['eigenvectors']
    
    if evals is not None and evecs is not None:
        stability_text += f"  Eigenvalues:\n   [{evals[0]:.4e}, {evals[1]:.4e}]\n\n"
        stability_text += f"  Eigenvectors:\n   v1: [{evecs[0,0]:.4f}, {evecs[1,0]:.4f}]\n"
        stability_text += f"   v2: [{evecs[0,1]:.4f}, {evecs[1,1]:.4f}]\n"
    else:
        stability_text += "  [Data missing]\n"
    
    info_ax.text(0, 0.9, psai_text + eq_text + stability_text, fontfamily='monospace', 
                fontsize=10, verticalalignment='top', horizontalalignment='left')
    
    if block:
        plt.show()
    else:
        plt.show(block=False)
        plt.pause(0.001)
        if display_seconds is not None:
            plt.pause(display_seconds)

    return fig


def plot_mode_2(X, Y, Fx, Fy, U_pot, source_positions, desired_pos, draw_desired_point=True):
    """Mode 2: Force field quiver on the left, potential well contour on the right."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Force Field (Normalized Quiver)
    F_mag = np.hypot(Fx, Fy)
    F_mag[F_mag == 0] = 1e-10 
    Fx_norm = Fx / F_mag
    Fy_norm = Fy / F_mag

    step = 3 
    ax1.quiver(X[::step, ::step], Y[::step, ::step], Fx_norm[::step, ::step], Fy_norm[::step, ::step], 
               color='black', pivot='mid', scale=30)
    ax1.plot(source_positions[:, 0], source_positions[:, 1], 'r.', markersize=15, label='Magnets')
    if draw_desired_point:
        ax1.plot(desired_pos[0], desired_pos[1], 'rx', markersize=15, markeredgewidth=2)
    ax1.set_title('Force Field Direction')
    ax1.set_aspect('equal')
    ax1.grid(True)
    
    # Right: Potential Energy (Clamped for better visibility)
    # Clamp the bottom 5% and top 80% to ignore the extreme singularities near magnets
    U_min = np.nanpercentile(U_pot, 5)
    U_max = np.nanpercentile(U_pot, 80)
    if U_min >= U_max:
        U_min = np.nanmin(U_pot)
        U_max = np.nanmax(U_pot)
        # If it's still completely flat, artificially bump U_max to prevent the crash
        if U_min == U_max:
            U_max = U_min + 1e-9
    levels = np.linspace(U_min, U_max, 50)
    
    cp = ax2.contourf(X, Y, U_pot, levels=levels, cmap='viridis', extend='both')
    fig.colorbar(cp, ax=ax2, label='Potential Energy (J)')
    
    ax2.plot(source_positions[:, 0], source_positions[:, 1], 'r.', markersize=15)
    if draw_desired_point:
        ax2.plot(desired_pos[0], desired_pos[1], 'rx', markersize=15, markeredgewidth=2)
    ax2.set_title('Potential Energy Well (Focused)')
    ax2.set_aspect('equal')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_mode_3(X, Y, Fx, Fy, Bx, By, source_positions, desired_pos, draw_desired_point=True):
    """Mode 3: Force field on the left, Rich Magnetic Field Heatmap on the right."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Force Field Direction
    F_mag = np.hypot(Fx, Fy)
    F_mag[F_mag == 0] = 1e-10
    Fx_norm, Fy_norm = Fx / F_mag, Fy / F_mag
    
    step = 3 
    ax1.quiver(X[::step, ::step], Y[::step, ::step], Fx_norm[::step, ::step], Fy_norm[::step, ::step], 
               color='black', pivot='mid', scale=30)
    ax1.plot(source_positions[:, 0], source_positions[:, 1], 'r.', markersize=15)
    if draw_desired_point:
        ax1.plot(desired_pos[0], desired_pos[1], 'rx', markersize=15, markeredgewidth=2)
    ax1.set_title('Force Field Direction')
    ax1.set_aspect('equal')
    ax1.grid(True)
    
    # Right: Magnetic Field Magnitude Heatmap + Streamlines
    B_mag = np.hypot(Bx, By)
    B_min = max(np.min(B_mag), 1e-10)
    B_max = np.percentile(B_mag, 95) # Cap max to avoid singularity blinding
    
    # Create log-spaced levels for the background color
    levels = np.logspace(np.log10(B_min), np.log10(B_max), 50)
    
    # Background: Log-scaled magnitude
    cp = ax2.contourf(X, Y, B_mag, levels=levels, norm=LogNorm(vmin=B_min, vmax=B_max), 
                      cmap='plasma', extend='max')
    fig.colorbar(cp, ax=ax2, label='Magnetic Field Magnitude |B| (Tesla)')
    
    # Foreground: White streamlines showing direction
    ax2.streamplot(X, Y, Bx, By, color='white', linewidth=1.0, density=1.5, arrowstyle='->')
    
    ax2.plot(source_positions[:, 0], source_positions[:, 1], 'k.', markersize=15) # Black dots for magnets over bright heatmap
    if draw_desired_point:
        ax2.plot(desired_pos[0], desired_pos[1], 'wx', markersize=15, markeredgewidth=2) # White X
    ax2.set_title('Magnetic Field: Magnitude & Direction')
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()



def save_temp_plot(fig, idx, base_dir="outputs", folder_name="temp_plots", dpi=200):
    """Save a Matplotlib figure into a temporary plots folder."""
    if fig is None:
        raise ValueError("save_temp_plot received fig=None. Check that plot_mode_1 returns fig.")

    temp_plot_dir = (Path.cwd() / base_dir / folder_name).resolve()
    temp_plot_dir.mkdir(parents=True, exist_ok=True)

    plot_filename = temp_plot_dir / f"plot_{idx:03d}.png"
    fig.savefig(plot_filename, dpi=dpi, bbox_inches="tight")

    print(f"Saved plot: {plot_filename}")
    return plot_filename




# def animate_trajectories(t_eval, trajectories, source_positions, target_schedule, grid_min, grid_max, save_video=False):
#     """
#     Creates an animation of the microrobots moving over time.
#     trajectories shape: (4*N, num_time_steps)
#     """
#     print("Generating animation...")
#     fig, ax = plt.subplots(figsize=(8, 8))
    
#     # Setup plot limits
#     ax.set_xlim(grid_min, grid_max)
#     ax.set_ylim(grid_min, grid_max)
#     ax.set_aspect('equal')
#     ax.grid(True, linestyle='--', alpha=0.6)
#     ax.set_title("Microrobot Swarm Dynamics")
#     ax.set_xlabel("X (m)")
#     ax.set_ylabel("Y (m)")
    
#     # Plot sources
#     sx = [p[0] for p in source_positions]
#     sy = [p[1] for p in source_positions]
#     ax.scatter(sx, sy, c='gray', s=100, marker='s', label='Magnets')
    
#     # Plot desired position
#     # ax.scatter(desired_pos[0], desired_pos[1], c='red', s=150, marker='X', label='Target')
#     initial_target = target_schedule[0][1]
#     target_scat = ax.scatter(
#         initial_target[0],
#         initial_target[1],
#         c='red',
#         s=150,
#         marker='X',
#         label='Active Target'
#     )
    
#     # Optional: show all scheduled targets faintly
#     all_targets = np.array([target_pos for _, target_pos in target_schedule])
#     ax.scatter(
#         all_targets[:, 0],
#         all_targets[:, 1],
#         c='red',
#         s=50,
#         marker='x',
#         alpha=0.4,
#         label='Scheduled Targets'
#     )
    
#     # Initialize scatter plot for robots
#     num_robots = len(trajectories) // 4
#     scat = ax.scatter([], [], c='blue', s=50, label='Microrobots')
#     ax.legend()
    
#     time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
#     def update(frame):
#         # Extract x and y positions for all robots at the current frame
#         x_pos = [trajectories[i*4, frame] for i in range(num_robots)]
#         y_pos = [trajectories[i*4+1, frame] for i in range(num_robots)]
#         scat.set_offsets(np.c_[x_pos, y_pos])
        
#         # ax.set_title(f"Microrobot Swarm Dynamics - Time: {t_eval[frame]:.2f}s")
#         time_text.set_text(f"Time: {t_eval[frame]:.2f} s")
        
#         return scat, time_text
    
#     ani = animation.FuncAnimation(fig, update, frames=len(t_eval), interval=30, blit=True)
    
#     if save_video:
#         print("Saving video to 'microrobots_simulation.mp4'...")
#         # Requires ffmpeg installed on your system
#         ani.save('microrobots_simulation.mp4', writer='ffmpeg', fps=30)
#         print("Video saved!")
    
#     plt.show() # Show the movie on screen
