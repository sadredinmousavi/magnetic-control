import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.constants import mu_0

from case_loader import unpack_target_schedule_entry

from functions_main import (
    calculate_capillary_force,
    calculate_dipole_interaction_force,
    calculate_robot_payload_interaction_force,
    calculate_total_force_from_sources,
)
from functions_utility import get_control_at_time, get_schedule_index


def get_target_points_from_schedule_entry(entry):
    _, target, additional, _, _ = unpack_target_schedule_entry(entry)
    if additional is None:
        return np.array([target])
    if isinstance(additional, list):
        return np.array([target, *additional])
    return np.array([target, additional])


def calculate_external_forces_batch(
    robot_positions,
    source_positions,
    control_inputs_u,
    m_source,
    m_robot
):
    r_vec = robot_positions[:, None, :] - source_positions[None, :, :]
    r_mag = np.linalg.norm(r_vec, axis=2)
    valid = r_mag > 1e-9

    c_f = (3 * mu_0 * m_source * m_robot) / (4 * np.pi)
    scale = np.zeros_like(r_mag)
    scale[valid] = (
        np.broadcast_to(control_inputs_u, r_mag.shape)[valid]
        / (r_mag[valid] ** 5)
    )

    return c_f * np.sum(r_vec * scale[:, :, None], axis=1)


def calculate_robot_robot_forces_batch(
    robot_positions,
    m_robot,
    robot_radius,
    gamma,
    capillary_sin_C
):
    r_vec = robot_positions[:, None, :] - robot_positions[None, :, :]
    r_mag = np.linalg.norm(r_vec, axis=2)
    valid = r_mag > 1e-9

    magnetic_scale = np.zeros_like(r_mag)
    magnetic_valid = r_mag > 1e-6
    magnetic_scale[magnetic_valid] = 1.0 / (r_mag[magnetic_valid] ** 5)
    k_mag = (3 * mu_0 * m_robot * m_robot) / (4 * np.pi)
    magnetic_forces = k_mag * np.sum(r_vec * magnetic_scale[:, :, None], axis=1)

    capillary_scale = np.zeros_like(r_mag)
    capillary_scale[valid] = 1.0 / (r_mag[valid] ** 2)
    k_cap = 2 * np.pi * gamma * robot_radius**2 * capillary_sin_C**2
    capillary_forces = k_cap * np.sum(r_vec * capillary_scale[:, :, None], axis=1)

    return magnetic_forces + capillary_forces


def calculate_wall_forces_batch(
    robot_positions,
    robot_velocities,
    wall_segments,
    robot_radius,
    wall_stiffness,
    wall_damping,
    wall_interaction_range
):
    if not wall_segments or wall_stiffness <= 0:
        return np.zeros_like(robot_positions)

    wall_forces = np.zeros_like(robot_positions)
    activation_distance = robot_radius + wall_interaction_range

    for wall_start, wall_end in wall_segments:
        start = np.asarray(wall_start, dtype=float)
        end = np.asarray(wall_end, dtype=float)
        segment = end - start
        segment_len_sq = np.dot(segment, segment)

        if segment_len_sq <= 1e-18:
            continue

        rel = robot_positions - start
        t = np.clip((rel @ segment) / segment_len_sq, 0.0, 1.0)
        closest = start + t[:, None] * segment
        offset = robot_positions - closest
        dist = np.linalg.norm(offset, axis=1)
        active = dist < activation_distance

        if not np.any(active):
            continue

        normals = np.zeros_like(offset)
        safe = active & (dist > 1e-12)
        normals[safe] = offset[safe] / dist[safe, None]

        degenerate = active & ~safe
        if np.any(degenerate):
            tangent = segment / np.sqrt(segment_len_sq)
            fallback_normal = np.array([-tangent[1], tangent[0]])
            normals[degenerate] = fallback_normal

        penetration = activation_distance - dist[active]
        normal_velocity = np.sum(robot_velocities[active] * normals[active], axis=1)
        damping_force = -wall_damping * np.minimum(normal_velocity, 0.0)
        force_magnitude = wall_stiffness * penetration + damping_force
        wall_forces[active] += force_magnitude[:, None] * normals[active]

    return wall_forces


def microrobot_payload_dynamics(
    t, state,
    source_positions,
    target_controls,
    M_SOURCE_MAGNITUDE,
    M_ROBOT_MAGNITUDE,
    robot_mass,
    fluid_drag,
    robot_radius,
    capillary_sin_C,
    gamma,
    payload_radius,
    payload_mass,
    payload_drag,
    contact_stiffness,
    contact_damping,
    payload_capillary_gain=0.0,
    payload_capillary_range=0.003,
    payload_capillary_cutoff=0.009,
    use_overdamped_dynamics=False,
    dynamics_speedup=1.0,
    wall_segments=None,
    wall_stiffness=0.0,
    wall_damping=0.0,
    wall_interaction_range=0.0
):
    """
    State layout:
    robots:
        [x1, y1, vx1, vy1, ..., xN, yN, vxN, vyN]
    payload:
        [xp, yp, vxp, vyp]
    """
    num_robot_states = len(state) - 4
    N = num_robot_states // 4

    derivatives = np.zeros_like(state)

    active_target, control_inputs_u = get_control_at_time(t, target_controls)

    payload_idx = 4 * N
    payload_pos = np.array([state[payload_idx], state[payload_idx + 1]])
    payload_vel = np.array([state[payload_idx + 2], state[payload_idx + 3]])

    F_payload = np.zeros(2)
    robot_states = state[:num_robot_states].reshape(N, 4)
    robot_positions = robot_states[:, :2]
    robot_velocities = robot_states[:, 2:4]

    external_forces = calculate_external_forces_batch(
        robot_positions,
        np.asarray(source_positions),
        np.asarray(control_inputs_u),
        M_SOURCE_MAGNITUDE,
        M_ROBOT_MAGNITUDE
    )
    robot_robot_forces = calculate_robot_robot_forces_batch(
        robot_positions,
        M_ROBOT_MAGNITUDE,
        robot_radius,
        gamma,
        capillary_sin_C
    )
    wall_forces = calculate_wall_forces_batch(
        robot_positions,
        robot_velocities,
        wall_segments or [],
        robot_radius,
        wall_stiffness,
        wall_damping,
        wall_interaction_range
    )

    for i in range(N):
        idx = i * 4
        pos_i = robot_positions[i]
        vel_i = robot_velocities[i]

        # 1. External magnetic force
        F_ext = external_forces[i]

        # 2. Robot-robot magnetic + capillary forces
        F_rr = robot_robot_forces[i]

        # # 3. Robot-payload contact
        # F_robot_on_payload_contact = calculate_robot_payload_contact_force(
        #     robot_pos=pos_i,
        #     robot_vel=vel_i,
        #     payload_pos=payload_pos,
        #     payload_vel=payload_vel,
        #     robot_radius=robot_radius,
        #     payload_radius=payload_radius,
        #     k_contact=contact_stiffness,
        #     c_contact=contact_damping
        # )

        # # Force on robot is opposite
        # F_payload_contact_on_robot = -F_robot_on_payload_contact

        # # 4. Robot-payload capillary attraction
        # F_robot_on_payload_cap = calculate_robot_payload_capillary_force(
        #     robot_pos=pos_i,
        #     payload_pos=payload_pos,
        #     robot_radius=robot_radius,
        #     payload_radius=payload_radius,
        #     capillary_gain=payload_capillary_gain,
        #     capillary_range=payload_capillary_range
        # )

        # # Attractive force on robot is opposite
        # F_payload_cap_on_robot = -F_robot_on_payload_cap

        # # Accumulate payload forces
        # F_payload += F_robot_on_payload_contact + F_robot_on_payload_cap

        # 3&4 Robot-payload contact and capillary attraction

        F_robot_on_payload = calculate_robot_payload_interaction_force(
            robot_pos=pos_i,
            robot_vel=vel_i,
            payload_pos=payload_pos,
            payload_vel=payload_vel,
            robot_radius=robot_radius,
            payload_radius=payload_radius,
            k_contact=contact_stiffness,
            c_contact=contact_damping,
            capillary_gain=payload_capillary_gain,
            capillary_range=payload_capillary_range,
            capillary_cutoff=payload_capillary_cutoff,
            adhesion_gap=0.0
        )

        F_payload_on_robot = -F_robot_on_payload

        F_payload += F_robot_on_payload


        F_robot_no_drag = (
            F_ext
            + F_rr
            + F_payload_on_robot
            + wall_forces[i]
        )

        if use_overdamped_dynamics:
            terminal_velocity = dynamics_speedup * F_robot_no_drag / fluid_drag
            derivatives[idx] = terminal_velocity[0]
            derivatives[idx + 1] = terminal_velocity[1]
            derivatives[idx + 2] = 0.0
            derivatives[idx + 3] = 0.0
            continue

        # 5. Robot drag
        F_drag = -fluid_drag * vel_i

        # 6. Robot net force
        F_robot = F_robot_no_drag + F_drag
        accel_i = F_robot / robot_mass

        derivatives[idx] = vel_i[0]
        derivatives[idx + 1] = vel_i[1]
        derivatives[idx + 2] = accel_i[0]
        derivatives[idx + 3] = accel_i[1]

    if use_overdamped_dynamics:
        payload_terminal_velocity = np.zeros(2)
        if payload_drag is not None and payload_drag > 0:
            payload_terminal_velocity = dynamics_speedup * F_payload / payload_drag

        derivatives[payload_idx] = payload_terminal_velocity[0]
        derivatives[payload_idx + 1] = payload_terminal_velocity[1]
        derivatives[payload_idx + 2] = 0.0
        derivatives[payload_idx + 3] = 0.0
        return derivatives

    # Payload dynamics
    F_payload_drag = -payload_drag * payload_vel
    F_payload_net = F_payload + F_payload_drag
    payload_accel = F_payload_net / payload_mass

    derivatives[payload_idx] = payload_vel[0]
    derivatives[payload_idx + 1] = payload_vel[1]
    derivatives[payload_idx + 2] = payload_accel[0]
    derivatives[payload_idx + 3] = payload_accel[1]

    # if int(t * 10) % 10 == 0:
    #     print(
    #         f"t={t:.2f}, "
    #         f"|F_payload|={np.linalg.norm(F_payload):.3e}, "
    #         f"|F_drag_payload|={np.linalg.norm(F_payload_drag):.3e}, "
    #         f"|a_payload|={np.linalg.norm(payload_accel):.3e}"
    #     )

    return derivatives






















def animate_trajectories(
    t_eval,
    trajectories,
    source_positions,
    target_schedule,
    grid_min,
    grid_max,
    field_data=None,
    draw_contour=True,
    draw_streamlines=True,
    draw_quiver=False,
    draw_sources=True,
    draw_all_targets=True,
    draw_active_target=True,
    plot_trajectories=True,
    plot_microrobots=True,
    payload_radius=None,
    wall_segments=None,
    contour_levels=20,
    save_video=False,
    video_name="microrobots_simulation.mp4",
    video_dpi=160,
    video_fps=30,
    video_crf=18,
    figure_size=(8, 8),
):
    """
    Fast animation for time-varying target microrobot simulation.

    Background fields are pre-drawn once for each target, then visibility is
    switched when the active target changes. This avoids redrawing contours
    every frame.
    """
    if video_dpi <= 0 or video_fps <= 0:
        raise ValueError("VIDEO_DPI and VIDEO_FPS must be positive.")
    if not 0 <= video_crf <= 51:
        raise ValueError("VIDEO_CRF must be between 0 and 51.")
    if len(figure_size) != 2 or any(size <= 0 for size in figure_size):
        raise ValueError("ANIMATION_FIGURE_SIZE must contain two positive values.")
    pixel_size = tuple(round(size * video_dpi) for size in figure_size)
    if any(pixels % 2 for pixels in pixel_size):
        raise ValueError(
            "ANIMATION_FIGURE_SIZE * VIDEO_DPI must produce even pixel dimensions "
            "for H.264 yuv420p output."
        )

    print("Generating animation...")

    fig, ax = plt.subplots(figsize=figure_size)
    # Keep the axes geometry identical between the interactive canvas and the
    # encoded frames. Explicit margins also prevent legends/labels from making
    # the saved axes box subtly narrower than it is on screen.
    fig.subplots_adjust(left=0.11, right=0.96, bottom=0.09, top=0.93)

    ax.set_xlim(grid_min, grid_max)
    ax.set_ylim(grid_min, grid_max)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title("Microrobot Swarm Dynamics")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    num_bodies = trajectories.shape[0] // 4
    has_payload = payload_radius is not None
    num_robots = num_bodies - 1 if has_payload else num_bodies

    # ---------------------------------------------------------
    # Helper: active target / field index
    # ---------------------------------------------------------
    def set_artist_visible(artists, visible):
        for artist in artists:
            if hasattr(artist, "set_visible"):
                artist.set_visible(visible)
            elif hasattr(artist, "collections"):
                for child in artist.collections:
                    child.set_visible(visible)

    # ---------------------------------------------------------
    # Pre-draw background layers
    # ---------------------------------------------------------
    background_layers = []

    if field_data is not None:
        for field in field_data:
            layer_artists = []

            X = field["X"]
            Y = field["Y"]
            Fx = field["Fx"]
            Fy = field["Fy"]
            U_pot = field["U_pot"]

            if draw_contour:
                F_mag = np.sqrt(Fx**2 + Fy**2)

                finite_vals = F_mag[np.isfinite(F_mag)]
                finite_vals = finite_vals[finite_vals > 0]

                if finite_vals.size == 0:
                    custom_levels = None
                else:
                    min_val = max(np.min(finite_vals), 1e-12)
                    max_val = np.percentile(finite_vals, 55)

                    if max_val <= min_val:
                        max_val = np.max(finite_vals)

                    if max_val <= min_val:
                        custom_levels = None
                    else:
                        custom_levels = np.logspace(
                            np.log10(min_val),
                            np.log10(max_val),
                            num=contour_levels
                        )

                contour = ax.contourf(
                    X,
                    Y,
                    F_mag,
                    levels=custom_levels,
                    cmap="viridis",
                    extend="both",
                    alpha=0.6
                )
                layer_artists.append(contour)

            if draw_streamlines:
                stream = ax.streamplot(
                    X,
                    Y,
                    Fx,
                    Fy,
                    density=1.0,
                    linewidth=0.7,
                    arrowsize=0.9,
                    color="black"
                )
                layer_artists.append(stream.lines)
                layer_artists.append(stream.arrows)

            if draw_quiver:
                skip = max(1, X.shape[0] // 20)
                q = ax.quiver(
                    X[::skip, ::skip],
                    Y[::skip, ::skip],
                    Fx[::skip, ::skip],
                    Fy[::skip, ::skip],
                    alpha=0.7
                )
                layer_artists.append(q)

            set_artist_visible(layer_artists, False)

            background_layers.append(layer_artists)

    # Show first background layer
    active_background_idx = {"idx": 0}

    if background_layers:
        set_artist_visible(background_layers[0], True)

    def set_background_layer(active_idx):
        if not background_layers:
            return

        if active_idx == active_background_idx["idx"]:
            return

        old_idx = active_background_idx["idx"]

        set_artist_visible(background_layers[old_idx], False)
        set_artist_visible(background_layers[active_idx], True)

        active_background_idx["idx"] = active_idx

    # ---------------------------------------------------------
    # Source magnets
    # ---------------------------------------------------------
    if draw_sources:
        source_positions = np.asarray(source_positions)
        ax.scatter(
            source_positions[:, 0],
            source_positions[:, 1],
            c="gray",
            s=100,
            marker="s",
            edgecolors="black",
            label="Source magnets"
        )

    if wall_segments:
        for idx, (wall_start, wall_end) in enumerate(wall_segments):
            wall_start = np.asarray(wall_start)
            wall_end = np.asarray(wall_end)
            ax.plot(
                [wall_start[0], wall_end[0]],
                [wall_start[1], wall_end[1]],
                color="dimgray",
                linewidth=4,
                solid_capstyle="round",
                label="Walls" if idx == 0 else None
            )

    # ---------------------------------------------------------
    # Scheduled targets
    # ---------------------------------------------------------
    if draw_all_targets:
        all_targets = np.vstack([
            get_target_points_from_schedule_entry(entry)
            for entry in target_schedule
        ])
        ax.scatter(
            all_targets[:, 0],
            all_targets[:, 1],
            c="red",
            s=60,
            marker="x",
            alpha=0.5,
            label="Scheduled targets"
        )

    # ---------------------------------------------------------
    # Active target
    # ---------------------------------------------------------
    active_target_scat = None
    if draw_active_target:
        initial_targets = get_target_points_from_schedule_entry(target_schedule[0])
        active_target_scat = ax.scatter(
            initial_targets[:, 0],
            initial_targets[:, 1],
            c="red",
            s=180,
            marker="X",
            edgecolors="black",
            label="Active target"
        )

    # ---------------------------------------------------------
    # Microrobots
    # ---------------------------------------------------------
    robot_scat = None
    if plot_microrobots:
        robot_scat = ax.scatter(
            [],
            [],
            c="blue",
            s=55,
            marker="o",
            edgecolors="black",
            label="Microrobots"
        )

    # ---------------------------------------------------------
    # Payload
    # ---------------------------------------------------------
    payload_patch = None
    if has_payload:
        payload_patch = plt.Circle(
            (0.0, 0.0),
            payload_radius,
            color="orange",
            alpha=0.65,
            ec="black",
            linewidth=1.5,
            label="Payload",
            zorder=6
        )
        ax.add_patch(payload_patch)
    
    # ---------------------------------------------------------
    # Trajectory lines
    # ---------------------------------------------------------
    trajectory_lines = []
    if plot_trajectories:
        for _ in range(num_robots):
            line, = ax.plot(
                [],
                [],
                linewidth=1.5,
                alpha=0.8
            )
            trajectory_lines.append(line)

    # ---------------------------------------------------------
    # Time text
    # ---------------------------------------------------------
    time_text = ax.text(
        0.04,
        0.96,
        "",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    ax.legend(loc="upper right")

    # ---------------------------------------------------------
    # Update function
    # ---------------------------------------------------------
    def update(frame):
        current_time = t_eval[frame]
        active_idx = get_schedule_index(current_time, target_schedule)

        set_background_layer(active_idx)

        x_pos = np.array([
            trajectories[i * 4, frame]
            for i in range(num_robots)
        ])
        y_pos = np.array([
            trajectories[i * 4 + 1, frame]
            for i in range(num_robots)
        ])

        artists = []

        if plot_microrobots and robot_scat is not None:
            robot_scat.set_offsets(np.c_[x_pos, y_pos])
            artists.append(robot_scat)

        if plot_trajectories:
            for i, line in enumerate(trajectory_lines):
                x_hist = trajectories[i * 4, :frame + 1]
                y_hist = trajectories[i * 4 + 1, :frame + 1]
                line.set_data(x_hist, y_hist)
                artists.append(line)

        if draw_active_target and active_target_scat is not None:
            active_targets = get_target_points_from_schedule_entry(target_schedule[active_idx])
            active_target_scat.set_offsets(active_targets)
            artists.append(active_target_scat)
        
        if has_payload and payload_patch is not None:
            payload_idx = num_robots * 4
            payload_x = trajectories[payload_idx, frame]
            payload_y = trajectories[payload_idx + 1, frame]
            payload_patch.center = (payload_x, payload_y)
            artists.append(payload_patch)

        time_text.set_text(
            f"Time: {current_time:.2f} s | Target {active_idx + 1}"
        )
        artists.append(time_text)

        if background_layers:
            artists.extend(background_layers[active_idx])

        return artists

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(t_eval),
        # Use the same nominal timing for the GUI preview and encoded file.
        interval=1000.0 / video_fps,
        blit=True,
        cache_frame_data=False,
    )

    if save_video:
        print(f"Saving video to '{video_name}'...")
        writer = animation.FFMpegWriter(
            fps=video_fps,
            codec="libx264",
            bitrate=-1,
            extra_args=[
                "-crf", str(video_crf),
                "-preset", "slow",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
            ],
            metadata={"title": "Microrobot Swarm Dynamics"},
        )
        ani.save(video_name, writer=writer, dpi=video_dpi)
        print("Video saved!")

    plt.show()
    return ani
