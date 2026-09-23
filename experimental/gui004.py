"""Offline video robot detection GUI."""

import csv
import json
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import tempfile
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
from scipy.optimize import linear_sum_assignment

from robot_vision import RobotDetector


PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_DIR / "inputs"
VIDEO_FILETYPES = [
    ("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v"),
    ("All files", "*.*"),
]
POST_ROBOT_MARKER_OPTIONS = ("Plus (+)", "Hollow rectangle")


def calibration_file_for(video_path):
    """Return the per-video calibration file path."""
    return INPUT_DIR / f"{Path(video_path).stem}_camera_calibration.json"


def robot_center_for(detections, expected_count=None):
    """Equal-weight position centroid; no temporal filtering or area weights."""
    if not detections or (expected_count is not None
                          and len(detections) != expected_count):
        return None
    centers = np.array([item["center"] for item in detections], dtype=float)
    if not np.isfinite(centers).all():
        return None
    return tuple(centers.mean(axis=0))


def match_robot_candidates(candidates, reference_points, max_distance):
    """Globally match blobs to robot references while preserving identities."""
    references = [tuple(map(float, point)) for point in reference_points]
    if not references or not candidates:
        return []

    candidate_centers = np.asarray(
        [candidate["center"] for candidate in candidates], dtype=float
    )
    reference_centers = np.asarray(references, dtype=float)
    distances = np.linalg.norm(
        reference_centers[:, None, :] - candidate_centers[None, :, :], axis=2
    )
    # A private dummy column for every robot permits a scientifically explicit
    # "unmatched" result instead of forcing an implausibly distant assignment.
    unmatched_cost = float(max_distance) + 1e-6
    invalid_cost = unmatched_cost + max(float(max_distance), 1.0) + 1.0
    candidate_costs = np.where(distances <= max_distance, distances, invalid_cost)
    dummy_costs = np.full(
        (len(references), len(references)), unmatched_cost, dtype=float
    )
    costs = np.concatenate((candidate_costs, dummy_costs), axis=1)
    reference_indices, column_indices = linear_sum_assignment(costs)

    matches = {}
    for reference_index, column_index in zip(reference_indices, column_indices):
        if (column_index < len(candidates)
                and distances[reference_index, column_index] <= max_distance):
            matches[reference_index] = (
                int(column_index), float(distances[reference_index, column_index])
            )

    selected = []
    for reference_index in range(len(references)):
        if reference_index not in matches:
            continue
        candidate_index, distance = matches[reference_index]
        detection = dict(candidates[candidate_index])
        detection["robot_id"] = reference_index + 1
        detection["candidate_index"] = int(
            candidates[candidate_index].get("candidate_index", candidate_index)
        )
        detection["match_distance_px"] = distance
        selected.append(detection)
    return selected


def exclude_calibration_markers(candidates, calibration_points, radius):
    """Separate candidates near fixed calibration dots without deleting data."""
    eligible = []
    excluded_indices = set()
    points = np.asarray(calibration_points, dtype=float)
    for candidate_index, candidate in enumerate(candidates):
        tagged = dict(candidate)
        tagged["candidate_index"] = candidate_index
        center = np.asarray(candidate["center"], dtype=float)
        near_marker = (
            radius > 0
            and len(points) > 0
            and float(np.linalg.norm(points - center, axis=1).min()) <= radius
        )
        if near_marker:
            excluded_indices.add(candidate_index)
        else:
            eligible.append(tagged)
    return eligible, excluded_indices


def draw_dashed_center_track(frame_bgr, center_track, current_frame,
                             color=(0, 0, 255)):
    annotated = frame_bgr.copy()
    dash_length, gap_length = 4.0, 3.0
    period = dash_length + gap_length
    distance_traveled = 0.0
    previous = None
    for frame_number, center in center_track:
        if frame_number > current_frame:
            break
        if center is None:
            previous = None
            distance_traveled = 0.0
            continue
        if previous is not None:
            previous_frame, previous_center = previous
            if frame_number == previous_frame + 1:
                start = np.array(previous_center, dtype=float)
                vector = np.array(center, dtype=float) - start
                length = float(np.linalg.norm(vector))
                position = 0.0
                while position < length:
                    phase = distance_traveled % period
                    remaining = min(length - position,
                                    (dash_length if phase < dash_length else period) - phase)
                    if phase < dash_length:
                        first = tuple(np.rint(start + vector * (position / length)).astype(int))
                        last = tuple(np.rint(start + vector * ((position + remaining) / length)).astype(int))
                        cv2.line(annotated, first, last, color, 1, cv2.LINE_AA)
                    position += remaining
                    distance_traveled += remaining
            else:
                distance_traveled = 0.0
        previous = (frame_number, center)
    return annotated


def select_cargo_candidate(candidates, previous_center=None, max_distance=100):
    """Select one red cargo component, independent of robot detections."""
    if not candidates:
        return None
    if previous_center is None:
        return dict(max(candidates, key=lambda item: float(item.get("area", 0))))
    nearby = []
    for candidate in candidates:
        distance = float(np.hypot(
            candidate["center"][0] - previous_center[0],
            candidate["center"][1] - previous_center[1],
        ))
        if distance <= max_distance:
            nearby.append((distance, -float(candidate.get("area", 0)), candidate))
    if not nearby:
        return None
    return dict(min(nearby, key=lambda item: (item[0], item[1]))[2])


def draw_cargo_overlay(frame_bgr, cargo_detection):
    """Draw the red cargo measurement in cyan so the red shape stays visible."""
    if not cargo_detection:
        return frame_bgr.copy()
    annotated = frame_bgr.copy()
    center = tuple(map(int, cargo_detection["center"]))
    bbox = cargo_detection.get("bbox")
    if bbox:
        x, y, width, height = map(int, bbox)
        cv2.rectangle(
            annotated, (x, y), (x + width, y + height),
            (255, 255, 0), 2, cv2.LINE_AA,
        )
    else:
        radius = max(8, int(round(np.sqrt(
            max(float(cargo_detection.get("area", 0)), 1.0) / np.pi
        ))))
        cv2.circle(annotated, center, radius, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.drawMarker(
        annotated, center, (255, 255, 0), cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA
    )
    cv2.putText(
        annotated, "Cargo", (center[0] + 13, center[1] - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA,
    )
    return annotated


def draw_detection_overlays(frame_bgr, detections, workspace_points=(),
                            show_robots=True, show_lines=True,
                            robot_marker_style="cross", expected_count=None):
    if robot_marker_style not in ("cross", "rectangle"):
        raise ValueError(
            "robot_marker_style must be either 'cross' or 'rectangle'."
        )
    annotated = frame_bgr.copy()
    if show_lines and len(workspace_points) == 4:
        cv2.polylines(annotated, [np.asarray(workspace_points, dtype=np.int32)],
                      True, (0, 255, 0), 1, cv2.LINE_AA)
    if show_robots and detections:
        for detection in detections:
            center = tuple(map(int, detection["center"]))
            if robot_marker_style == "cross":
                cv2.drawMarker(annotated, center, (0, 255, 0),
                               cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
            else:
                area = max(float(detection.get("area", 0) or 0), 0.0)
                half_size = 9 if area == 0 else int(np.clip(
                    round(np.sqrt(area) / 2.0 + 3.0), 7, 30
                ))
                x, y = center
                cv2.rectangle(
                    annotated,
                    (x - half_size, y - half_size),
                    (x + half_size, y + half_size),
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            if detection.get("robot_id") is not None:
                cv2.putText(
                    annotated,
                    f"R{detection['robot_id']}",
                    (center[0] + 11, center[1] - 9),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
        center = robot_center_for(detections, expected_count)
        if center is not None:
            cv2.drawMarker(annotated, tuple(np.rint(center).astype(int)), (0, 0, 255),
                           cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
    return annotated


def read_detections_csv(csv_path):
    by_frame = {}
    with Path(csv_path).open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            accepted = row.get("accepted", "").strip().lower()
            if accepted and accepted not in ("1", "true", "yes"):
                continue
            if not row.get("center_x_px") or not row.get("center_y_px"):
                continue
            frame = int(row["frame"])
            by_frame.setdefault(frame, []).append({
                "center": (int(row["center_x_px"]), int(row["center_y_px"])),
                "area": float(row["area_px2"] or 0),
                "robot_id": int(row["robot_id"]) if row.get("robot_id") else None,
            })
    return by_frame


def read_cargo_csv(csv_path):
    by_frame = {}
    path = Path(csv_path)
    if not path.is_file():
        return by_frame
    with path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            if not row.get("center_x_px") or not row.get("center_y_px"):
                continue
            detection = {
                "center": (int(row["center_x_px"]), int(row["center_y_px"])),
                "area": float(row.get("area_px2") or 0),
            }
            if all(row.get(name) for name in ("bbox_x", "bbox_y", "bbox_w", "bbox_h")):
                detection["bbox"] = tuple(
                    int(row[name]) for name in ("bbox_x", "bbox_y", "bbox_w", "bbox_h")
                )
            by_frame[int(row["frame"])] = detection
    return by_frame


def circle_detection_mask(shape, circle):
    if circle is None:
        return None
    x, y, radius = circle
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (((xx-x)**2 + (yy-y)**2 <= radius**2) * 255).astype(np.uint8)


def circular_video_frame(frame, center, radius, preview=False):
    """Mask after overlays; export an even-sized square for MP4 encoders."""
    radius = max(1, int(round(radius)))
    x, y = map(lambda value: int(round(value)), center)
    height, width = frame.shape[:2]
    yy, xx = np.ogrid[:height, :width]
    inside = (xx - x)**2 + (yy - y)**2 <= radius**2
    masked = frame.copy() if preview else np.zeros_like(frame)
    if preview:
        masked[~inside] = (frame[~inside] * 0.25).astype(np.uint8)
        cv2.circle(masked, (x, y), radius, (255, 255, 0), 1, cv2.LINE_AA)
        for dx, dy in ((radius, 0), (-radius, 0), (0, radius), (0, -radius)):
            cv2.circle(masked, (x + dx, y + dy), 4, (255, 255, 0), -1)
        return masked
    masked[inside] = frame[inside]
    output = np.zeros((2 * radius, 2 * radius, 3), dtype=frame.dtype)
    left, top = x - radius, y - radius
    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(width, x + radius), min(height, y + radius)
    if x1 > x0 and y1 > y0:
        output[y0-top:y1-top, x0-left:x1-left] = masked[y0:y1, x0:x1]
    return output


def annotation_bgra(annotation_bgr):
    """Convert annotations on black into straight-alpha BGRA pixels."""
    alpha = annotation_bgr.max(axis=2).astype(np.uint8)
    straight = np.zeros_like(annotation_bgr)
    visible = alpha > 0
    if np.any(visible):
        straight[visible] = np.clip(
            annotation_bgr[visible].astype(np.float32)
            * (255.0 / alpha[visible, None]),
            0,
            255,
        ).astype(np.uint8)
    return np.dstack((straight, alpha))


def annotation_on_chroma_color(annotation_bgr, color_bgr=(255, 0, 0)):
    """Composite antialiased annotations over a solid chroma-key color."""
    overlay = annotation_bgra(annotation_bgr)
    alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
    background = np.empty_like(annotation_bgr)
    background[:] = color_bgr
    return np.clip(
        overlay[:, :, :3].astype(np.float32) * alpha
        + background.astype(np.float32) * (1.0 - alpha),
        0,
        255,
    ).astype(np.uint8)


def render_annotation_frame(
    shape,
    frame_number,
    detections,
    center_track,
    workspace,
    show_robots,
    show_lines,
    show_center_track,
    marker_style,
    expected_count,
    cargo_detections,
    cargo_track,
    show_cargo,
    show_cargo_track,
):
    frame = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    if show_center_track:
        frame = draw_dashed_center_track(frame, center_track, frame_number)
    if show_cargo_track and cargo_track:
        frame = draw_dashed_center_track(
            frame, cargo_track, frame_number, color=(255, 255, 0)
        )
    frame = draw_detection_overlays(
        frame, detections.get(frame_number, []), workspace,
        show_robots, show_lines, marker_style, expected_count,
    )
    if show_cargo:
        frame = draw_cargo_overlay(frame, cargo_detections.get(frame_number))
    return frame


def export_annotation_overlay(
    source,
    destination,
    detections,
    center_track,
    workspace,
    show_robots,
    show_lines,
    show_center_track,
    marker_style,
    expected_count,
    cargo_detections,
    cargo_track,
    show_cargo,
    show_cargo_track,
    stop_event,
    updates,
):
    """Export a lossless alpha-only MOV matching the source video timing."""
    capture = cv2.VideoCapture(str(source))
    process = None
    temporary = None
    try:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise ValueError("FFmpeg was not found; transparent MOV export requires it.")
        if not capture.isOpened():
            raise ValueError("Could not open the original video.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if np.isfinite(fps) and fps > 0 else 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        suffix = Path(destination).suffix.lower()
        if suffix not in (".avi", ".mov"):
            raise ValueError("Choose .avi or .mov for transparent video export.")
        descriptor, temporary = tempfile.mkstemp(
            suffix=suffix, prefix=".annotation-overlay-",
            dir=Path(destination).parent,
        )
        os.close(descriptor)
        codec_options = (
            ["-c:v", "png", "-pix_fmt", "rgba"]
            if suffix == ".avi"
            else ["-c:v", "qtrle", "-pix_fmt", "argb"]
        )
        command = [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pixel_format", "bgra",
            "-video_size", f"{width}x{height}",
            "-framerate", f"{fps:.12g}", "-i", "-", "-an",
            *codec_options, temporary,
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        frame_number = 0
        while not stop_event.is_set():
            success, _ = capture.read()
            if not success:
                break
            frame = render_annotation_frame(
                (height, width), frame_number, detections, center_track,
                workspace, show_robots, show_lines, show_center_track,
                marker_style, expected_count, cargo_detections, cargo_track,
                show_cargo, show_cargo_track,
            )
            process.stdin.write(annotation_bgra(frame).tobytes())
            frame_number += 1
            if frame_number % 30 == 0:
                updates.put(("progress", frame_number, total))

        process.stdin.close()
        error_text = process.stderr.read().decode("utf-8", errors="replace").strip()
        return_code = process.wait()
        process = None
        if stop_event.is_set():
            updates.put(("cancelled",))
        elif return_code != 0:
            raise ValueError(error_text or "FFmpeg could not encode the overlay.")
        elif not frame_number or (total > 0 and frame_number < total):
            raise ValueError(
                f"Video ended unexpectedly after {frame_number} of {total} frames."
            )
        else:
            os.replace(temporary, destination)
            temporary = None
            updates.put(("done", str(destination), frame_number))
    except Exception as error:
        updates.put(("error", str(error)))
    finally:
        capture.release()
        if process is not None:
            if process.stdin:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            process.terminate()
            process.wait()
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def export_annotation_png_sequence(
    source,
    destination,
    detections,
    center_track,
    workspace,
    show_robots,
    show_lines,
    show_center_track,
    marker_style,
    expected_count,
    cargo_detections,
    cargo_track,
    show_cargo,
    show_cargo_track,
    stop_event,
    updates,
):
    """Export lossless transparent PNG frames for universal editor import."""
    capture = cv2.VideoCapture(str(source))
    destination = Path(destination)
    created_files = []
    try:
        if not capture.isOpened():
            raise ValueError("Could not open the original video.")
        if destination.exists() and any(destination.iterdir()):
            raise ValueError("Choose an empty folder for the PNG sequence.")
        destination.mkdir(parents=True, exist_ok=True)
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if np.isfinite(fps) and fps > 0 else 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_number = 0
        while not stop_event.is_set():
            success, _ = capture.read()
            if not success:
                break
            frame = render_annotation_frame(
                (height, width), frame_number, detections, center_track,
                workspace, show_robots, show_lines, show_center_track,
                marker_style, expected_count, cargo_detections, cargo_track,
                show_cargo, show_cargo_track,
            )
            frame_path = destination / f"annotations_{frame_number:08d}.png"
            if not cv2.imwrite(str(frame_path), annotation_bgra(frame)):
                raise ValueError(f"Could not write {frame_path.name}.")
            created_files.append(frame_path)
            frame_number += 1
            if frame_number % 30 == 0:
                updates.put(("progress", frame_number, total))
        if stop_event.is_set():
            for path in created_files:
                path.unlink(missing_ok=True)
            updates.put(("cancelled",))
        elif not frame_number or (total > 0 and frame_number < total):
            raise ValueError(
                f"Video ended unexpectedly after {frame_number} of {total} frames."
            )
        else:
            metadata = destination / "sequence_info.json"
            metadata.write_text(json.dumps({
                "fps": fps,
                "width": width,
                "height": height,
                "frame_count": frame_number,
                "first_frame": "annotations_00000000.png",
            }, indent=2) + "\n", encoding="utf-8")
            updates.put(("done", str(destination), frame_number))
    except Exception as error:
        for path in created_files:
            path.unlink(missing_ok=True)
        updates.put(("error", str(error)))
    finally:
        capture.release()


def export_chroma_annotation_video(
    source,
    destination,
    detections,
    center_track,
    workspace,
    show_robots,
    show_lines,
    show_center_track,
    marker_style,
    expected_count,
    cargo_detections,
    cargo_track,
    show_cargo,
    show_cargo_track,
    stop_event,
    updates,
    chroma_color_bgr=(255, 0, 0),
):
    """Export annotations over a solid blue background as high-quality H.264."""
    capture = cv2.VideoCapture(str(source))
    process = None
    temporary = None
    try:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise ValueError("FFmpeg was not found; MP4 export requires it.")
        if not capture.isOpened():
            raise ValueError("Could not open the original video.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if np.isfinite(fps) and fps > 0 else 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        descriptor, temporary = tempfile.mkstemp(
            suffix=".mp4", prefix=".annotation-chroma-",
            dir=Path(destination).parent,
        )
        os.close(descriptor)
        process = subprocess.Popen(
            [
                ffmpeg, "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pixel_format", "bgr24",
                "-video_size", f"{width}x{height}",
                "-framerate", f"{fps:.12g}", "-i", "-", "-an",
                "-c:v", "libx264", "-preset", "medium", "-crf", "8",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", temporary,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        frame_number = 0
        while not stop_event.is_set():
            success, _ = capture.read()
            if not success:
                break
            annotation = render_annotation_frame(
                (height, width), frame_number, detections, center_track,
                workspace, show_robots, show_lines, show_center_track,
                marker_style, expected_count, cargo_detections, cargo_track,
                show_cargo, show_cargo_track,
            )
            frame = annotation_on_chroma_color(annotation, chroma_color_bgr)
            process.stdin.write(frame.tobytes())
            frame_number += 1
            if frame_number % 30 == 0:
                updates.put(("progress", frame_number, total))
        process.stdin.close()
        error_text = process.stderr.read().decode("utf-8", errors="replace").strip()
        return_code = process.wait()
        process = None
        if stop_event.is_set():
            updates.put(("cancelled",))
        elif return_code != 0:
            raise ValueError(error_text or "FFmpeg could not encode the MP4 overlay.")
        elif not frame_number or (total > 0 and frame_number < total):
            raise ValueError(
                f"Video ended unexpectedly after {frame_number} of {total} frames."
            )
        else:
            os.replace(temporary, destination)
            temporary = None
            updates.put(("done", str(destination), frame_number))
    except Exception as error:
        updates.put(("error", str(error)))
    finally:
        capture.release()
        if process is not None:
            if process.stdin:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            process.terminate()
            process.wait()
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def export_circular_video(source, destination, center, radius, detections,
                          center_track, workspace, show_robots, show_lines,
                          show_track, marker_style, stop_event, updates,
                          expected_count=None, cargo_detections=None,
                          cargo_track=None, show_cargo=True,
                          show_cargo_track=True):
    """Render from original coordinates in a worker; publish only complete files."""
    capture = cv2.VideoCapture(str(source))
    writer = None
    temporary = None
    try:
        if not capture.isOpened():
            raise ValueError("Could not open the original video.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if np.isfinite(fps) and fps > 0 else 30.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        radius = max(1, int(round(radius)))
        descriptor, temporary = tempfile.mkstemp(
            suffix=".mp4", prefix=".circle-export-", dir=Path(destination).parent
        )
        os.close(descriptor)
        writer = cv2.VideoWriter(temporary, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (2 * radius, 2 * radius))
        if not writer.isOpened():
            raise ValueError("Could not initialize the MP4 encoder.")
        count = 0
        while not stop_event.is_set():
            success, frame = capture.read()
            if not success:
                break
            if show_track:
                frame = draw_dashed_center_track(frame, center_track, count)
            if show_cargo_track and cargo_track:
                frame = draw_dashed_center_track(
                    frame, cargo_track, count, color=(255, 255, 0)
                )
            frame = draw_detection_overlays(
                frame, detections.get(count, []), workspace,
                show_robots, show_lines, marker_style, expected_count,
            )
            if show_cargo and cargo_detections:
                frame = draw_cargo_overlay(frame, cargo_detections.get(count))
            writer.write(circular_video_frame(frame, center, radius))
            count += 1
            if count % 30 == 0:
                updates.put(("progress", count, total))
        writer.release()
        writer = None
        if stop_event.is_set():
            updates.put(("cancelled",))
        elif not count or (total > 0 and count < total):
            raise ValueError(f"Video ended unexpectedly after {count} of {total} frames.")
        else:
            os.replace(temporary, destination)
            temporary = None
            updates.put(("done", str(destination), count))
    except Exception as error:
        updates.put(("error", str(error)))
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


class OfflineDetectionGUI:
    """Process a recorded video as fast as possible and save its detections."""

    def __init__(self, root):
        self.root = root
        self.root.title("Offline Microrobot Detection")
        self.root.geometry("960x720")
        self.root.minsize(720, 540)

        self.detector = RobotDetector()
        self.stop_event = None
        self.worker_thread = None
        self.preview_queue = queue.Queue(maxsize=1)
        self.preview_photo = None
        self.preview_capture = None
        self.current_video_path = None
        self.playback_after_id = None
        self.video_playing = False
        self.video_fps = 30.0
        self.video_total_frames = 0
        self.current_frame_number = 0
        self.current_frame_bgr = None
        self.display_transform = None
        self.calibration_points = []
        self.robot_points = []
        self.current_cargo_detection = None
        self.learned_robot_color_ranges = None
        self.timeline_is_updating = False
        self.post_detections = {}
        self.post_center_track = []
        self.post_cargo_detections = {}
        self.post_cargo_track = []
        self.post_frame_number = 0
        self.post_playing = False
        self.post_after_id = None
        self.post_timeline_is_updating = False

        self.crop_center = None
        self.detection_circle = None
        self.detection_drag = None
        self.crop_radius = 1
        self.crop_drag = None
        self.crop_export_thread = None
        self.crop_export_stop = threading.Event()
        self.crop_export_updates = queue.Queue()
        self.overlay_export_thread = None
        self.overlay_export_stop = threading.Event()
        self.overlay_export_updates = queue.Queue()

        self.build_controls()
        self.build_viewer()
        self.root.after(50, self.poll_preview)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def build_controls(self):
        self.phase_notebook = ttk.Notebook(self.root)
        self.phase_notebook.pack(fill="x", padx=10, pady=(10, 4))
        self.phase_notebook.bind("<<NotebookTabChanged>>", self.on_phase_changed)

        calibration = tk.Frame(self.phase_notebook, padx=10, pady=8)
        robot_finding = tk.Frame(self.phase_notebook, padx=10, pady=8)
        processing = tk.Frame(self.phase_notebook, padx=10, pady=8)
        post_processing = tk.Frame(self.phase_notebook, padx=10, pady=8)
        self.phase_notebook.add(calibration, text="Phase 1 - Calibration")
        self.phase_notebook.add(robot_finding, text="Phase 2 - Find robots")
        self.phase_notebook.add(processing, text="Phase 3 - Process")
        self.phase_notebook.add(post_processing, text="Phase 4 - Post-process")
        calibration.columnconfigure(1, weight=1)
        robot_finding.columnconfigure(1, weight=1)
        processing.columnconfigure(1, weight=1)
        post_processing.columnconfigure(1, weight=1)

        tk.Label(calibration, text="Video file").grid(row=0, column=0, sticky="w")
        self.video_path_var = tk.StringVar()
        self.video_entry = tk.Entry(calibration, textvariable=self.video_path_var)
        self.video_entry.grid(
            row=0, column=1, columnspan=4, sticky="ew", padx=8, pady=4
        )
        self.browse_button = tk.Button(
            calibration, text="Browse...", command=self.choose_video
        )
        self.browse_button.grid(row=0, column=5, padx=4)

        self.play_button = tk.Button(
            calibration, text="Play", width=8, command=self.toggle_playback
        )
        self.play_button.grid(row=1, column=0, padx=4, pady=4)
        self.timeline = tk.Scale(
            calibration,
            from_=0,
            to=1,
            orient="horizontal",
            showvalue=False,
            command=self.on_timeline_changed,
        )
        self.timeline.grid(row=1, column=1, columnspan=4, sticky="ew", padx=8)
        self.time_label = tk.Label(calibration, text="00:00.000 / 00:00.000")
        self.time_label.grid(row=1, column=5, padx=4)

        tk.Label(calibration, text="h (cm)").grid(row=2, column=0, sticky="w")
        self.object_height_var = tk.StringVar(value="1")
        tk.Entry(
            calibration, textvariable=self.object_height_var, width=8
        ).grid(row=2, column=1, padx=8, pady=4, sticky="w")

        tk.Label(calibration, text="Camera to dot surface (cm)").grid(
            row=2, column=2, columnspan=2, sticky="e"
        )
        self.camera_height_var = tk.StringVar(value="29")
        tk.Entry(
            calibration, textvariable=self.camera_height_var, width=8
        ).grid(row=2, column=4, padx=8, pady=4, sticky="w")

        tk.Label(calibration, text="Dot rectangle width (cm)").grid(
            row=3, column=0, sticky="w"
        )
        self.rectangle_width_var = tk.StringVar(value="10")
        tk.Entry(
            calibration, textvariable=self.rectangle_width_var, width=8
        ).grid(row=3, column=1, padx=8, pady=4, sticky="w")

        tk.Label(calibration, text="Dot rectangle height (cm)").grid(
            row=3, column=2, columnspan=2, sticky="e"
        )
        self.rectangle_height_var = tk.StringVar(value="10")
        tk.Entry(
            calibration, textvariable=self.rectangle_height_var, width=8
        ).grid(row=3, column=4, padx=8, pady=4, sticky="w")

        tk.Label(calibration, text="Green-dot search radius (px)").grid(
            row=4, column=0, sticky="w"
        )
        self.calibration_search_radius_var = tk.StringVar(value="35")
        tk.Entry(
            calibration, textvariable=self.calibration_search_radius_var, width=8
        ).grid(row=4, column=1, padx=8, pady=4, sticky="w")

        tk.Label(calibration, text="Green-dot min area (px²)").grid(
            row=4, column=2, columnspan=2, sticky="e"
        )
        self.calibration_minimum_area_var = tk.StringVar(value="1")
        tk.Entry(
            calibration, textvariable=self.calibration_minimum_area_var, width=8
        ).grid(row=4, column=4, padx=8, pady=4, sticky="w")

        self.calibration_status_label = tk.Label(
            calibration,
            text="Choose a video, find a clear frame, then click the 4 green dots.",
            fg="blue",
            anchor="w",
        )
        self.calibration_status_label.grid(
            row=5, column=0, columnspan=4, sticky="ew", pady=4
        )
        tk.Button(
            calibration, text="Clear points", command=self.clear_calibration_points
        ).grid(row=5, column=4, padx=4)
        tk.Button(
            calibration, text="Save calibration", command=self.save_calibration
        ).grid(row=5, column=5, padx=4)

        tk.Label(robot_finding, text="Finding method").grid(
            row=0, column=0, sticky="w"
        )
        self.robot_finding_mode_var = tk.StringVar(value="Manual selection")
        tk.OptionMenu(
            robot_finding,
            self.robot_finding_mode_var,
            "Manual selection",
            "Color detection",
        ).grid(row=0, column=1, sticky="w", padx=8, pady=4)

        tk.Label(robot_finding, text="Robot color").grid(
            row=0, column=2, sticky="e"
        )
        self.detection_color_var = tk.StringVar(value="Learned from clicks")
        tk.OptionMenu(
            robot_finding,
            self.detection_color_var,
            "Learned from clicks",
            *RobotDetector.COLOR_RANGES,
        ).grid(row=0, column=3, sticky="w", padx=8, pady=4)

        tk.Label(robot_finding, text="Detection min area (px²)").grid(
            row=0, column=4, sticky="e"
        )
        self.minimum_area_var = tk.StringVar(value="20")
        tk.Entry(
            robot_finding, textvariable=self.minimum_area_var, width=8
        ).grid(row=0, column=5, padx=4, pady=4)

        tk.Label(robot_finding, text="Number of robots").grid(
            row=1, column=0, sticky="w"
        )
        self.robot_count_var = tk.StringVar(value="3")
        tk.Entry(
            robot_finding, textvariable=self.robot_count_var, width=8
        ).grid(row=1, column=1, padx=8, pady=4, sticky="w")
        tk.Label(robot_finding, text="Click/tracking radius (px)").grid(
            row=1, column=2, sticky="e"
        )
        self.robot_search_radius_var = tk.StringVar(value="35")
        tk.Entry(
            robot_finding, textvariable=self.robot_search_radius_var, width=8
        ).grid(row=1, column=3, padx=8, pady=4, sticky="w")
        tk.Label(robot_finding, text="Click min area (px²)").grid(
            row=1, column=4, sticky="e"
        )
        self.robot_snap_minimum_area_var = tk.StringVar(value="1")
        tk.Entry(
            robot_finding, textvariable=self.robot_snap_minimum_area_var, width=8
        ).grid(row=1, column=5, padx=4, pady=4, sticky="w")
        tk.Button(
            robot_finding, text="Find by color", command=self.find_robots_by_color
        ).grid(row=2, column=4, padx=4)
        tk.Button(
            robot_finding, text="Clear robots", command=self.clear_robot_points
        ).grid(row=2, column=5, padx=4)
        tk.Label(robot_finding, text="Detection cleanup size (px)").grid(
            row=2, column=0, sticky="w"
        )
        self.detection_filter_size_var = tk.StringVar(value="1")
        tk.OptionMenu(
            robot_finding, self.detection_filter_size_var, "1", "3", "5"
        ).grid(row=2, column=1, sticky="w", padx=8, pady=4)
        tk.Label(robot_finding, text="Calibration-dot exclusion (px)").grid(
            row=2, column=2, sticky="e"
        )
        self.calibration_exclusion_radius_var = tk.StringVar(value="12")
        tk.Entry(
            robot_finding,
            textvariable=self.calibration_exclusion_radius_var,
            width=8,
        ).grid(row=2, column=3, padx=8, pady=4, sticky="w")
        self.robot_status_label = tk.Label(
            robot_finding,
            text="Choose manual selection and click each robot, or find them by color.",
            fg="blue",
            anchor="w",
        )
        self.robot_status_label.grid(
            row=3, column=0, columnspan=6, sticky="ew", pady=4
        )

        cargo_controls = ttk.LabelFrame(robot_finding, text="Optional red cargo", padding=5)
        cargo_controls.grid(row=4, column=0, columnspan=6, sticky="ew", pady=(2, 4))
        self.cargo_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            cargo_controls, text="Detect red cargo", variable=self.cargo_enabled_var,
            command=self.on_cargo_toggle,
        ).grid(row=0, column=0, padx=4)
        ttk.Label(cargo_controls, text="Min area (px²)").grid(row=0, column=1)
        self.cargo_minimum_area_var = tk.StringVar(value="200")
        ttk.Entry(
            cargo_controls, textvariable=self.cargo_minimum_area_var, width=8
        ).grid(row=0, column=2, padx=4)
        ttk.Label(cargo_controls, text="Tracking radius (px)").grid(row=0, column=3)
        self.cargo_tracking_radius_var = tk.StringVar(value="100")
        ttk.Entry(
            cargo_controls, textvariable=self.cargo_tracking_radius_var, width=8
        ).grid(row=0, column=4, padx=4)
        ttk.Button(
            cargo_controls, text="Preview cargo", command=self.preview_cargo
        ).grid(row=0, column=5, padx=4)
        self.cargo_status_label = ttk.Label(
            cargo_controls, text="Disabled; robot detection is unchanged."
        )
        self.cargo_status_label.grid(row=1, column=0, columnspan=6, sticky="w", padx=4)

        region = ttk.LabelFrame(robot_finding, text="Circular detection area", padding=5)
        region.grid(row=5, column=0, columnspan=6, sticky="ew")
        region.columnconfigure(2, weight=1)
        self.detection_circle_enabled = tk.BooleanVar(value=False)
        self.detection_circle_edit = tk.BooleanVar(value=False)
        ttk.Checkbutton(region, text="Enable", variable=self.detection_circle_enabled,
                        command=self.render_calibration_frame).grid(row=0, column=0)
        ttk.Checkbutton(
            region,
            text="Adjust circle (blocks selection)",
            variable=self.detection_circle_edit,
            command=self.on_detection_edit_changed,
        ).grid(row=0, column=1)
        self.detection_diameter = tk.DoubleVar(value=2)
        self.detection_slider = ttk.Scale(region, from_=2, to=1000,
            variable=self.detection_diameter, command=self.resize_detection_circle)
        self.detection_slider.grid(row=0, column=2, sticky="ew", padx=8)
        self.detection_size_label = ttk.Label(region, text="2 px")
        self.detection_size_label.grid(row=0, column=3)
        ttk.Button(region, text="Fit to workspace", command=self.fit_detection_circle
                   ).grid(row=0, column=4)
        ttk.Label(region, text="Adjust circle: drag inside to move, edge to resize. Uncheck to select robots."
                  ).grid(row=1, column=0, columnspan=5, sticky="w")

        self.process_button = tk.Button(
            processing, text="Process Video", command=self.start_processing
        )
        self.process_button.grid(row=0, column=0, padx=4, pady=6, sticky="w")
        self.stop_button = tk.Button(
            processing, text="Stop", command=self.stop_processing, state="disabled"
        )
        self.stop_button.grid(row=0, column=1, padx=4, pady=6, sticky="w")

        self.progress = ttk.Progressbar(
            processing, orient="horizontal", mode="determinate", maximum=100
        )
        self.progress.grid(
            row=0, column=2, columnspan=4, sticky="ew", padx=8, pady=6
        )
        tk.Label(
            processing,
            text="Uses the Phase 2 color and enabled circular detection area.",
            anchor="w",
        ).grid(row=1, column=0, columnspan=6, sticky="ew")

        self.post_play_button = tk.Button(
            post_processing, text="Play", command=self.toggle_post_playback
        )
        self.post_play_button.grid(row=0, column=0, padx=4, pady=4)
        self.post_timeline = tk.Scale(
            post_processing, from_=0, to=1, orient="horizontal", showvalue=False,
            command=self.on_post_timeline_changed,
        )
        self.post_timeline.grid(row=0, column=1, columnspan=3, sticky="ew", padx=8)
        self.post_time_label = tk.Label(post_processing, text="00:00.000 / 00:00.000")
        self.post_time_label.grid(row=0, column=4, padx=4)
        tk.Button(post_processing, text="Load detections",
                  command=self.choose_post_detections).grid(row=0, column=5, padx=4)
        self.show_robot_marks_var = tk.BooleanVar(value=True)
        self.post_robot_marker_style_var = tk.StringVar(
            value=POST_ROBOT_MARKER_OPTIONS[0]
        )
        self.show_lines_var = tk.BooleanVar(value=True)
        self.show_center_track_var = tk.BooleanVar(value=False)
        self.show_cargo_var = tk.BooleanVar(value=True)
        self.show_cargo_track_var = tk.BooleanVar(value=True)
        tk.Checkbutton(post_processing, text="Robot marks",
                       variable=self.show_robot_marks_var,
                       command=self.render_post_frame).grid(row=1, column=0,
                                                            sticky="w")
        self.post_robot_marker_style = ttk.Combobox(
            post_processing,
            textvariable=self.post_robot_marker_style_var,
            values=POST_ROBOT_MARKER_OPTIONS,
            state="readonly",
            width=16,
        )
        self.post_robot_marker_style.grid(row=1, column=1, sticky="w", padx=(0, 8))
        self.post_robot_marker_style.bind(
            "<<ComboboxSelected>>", lambda _event: self.render_post_frame()
        )
        tk.Checkbutton(post_processing, text="Lines",
                       variable=self.show_lines_var,
                       command=self.render_post_frame).grid(row=1, column=2,
                                                            columnspan=2, sticky="w")
        tk.Checkbutton(post_processing, text="Red center track",
                       variable=self.show_center_track_var,
                       command=self.render_post_frame).grid(row=1, column=4,
                                                            columnspan=2, sticky="w")
        tk.Checkbutton(
            post_processing, text="Cargo mark", variable=self.show_cargo_var,
            command=self.render_post_frame,
        ).grid(row=2, column=0, columnspan=2, sticky="w")
        tk.Checkbutton(
            post_processing, text="Cyan cargo translation",
            variable=self.show_cargo_track_var, command=self.render_post_frame,
        ).grid(row=2, column=2, columnspan=3, sticky="w")
        self.post_cargo_status_label = ttk.Label(
            post_processing,
            text="Cargo data: none loaded. Enable it in Phase 2 and process the video.",
        )
        self.post_cargo_status_label.grid(
            row=3, column=0, columnspan=6, sticky="w"
        )

        crop_controls = ttk.LabelFrame(post_processing, text="Circular crop", padding=6)
        crop_controls.grid(row=4, column=0, columnspan=6, sticky="ew", pady=(6, 0))
        crop_controls.columnconfigure(2, weight=1)
        self.crop_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(crop_controls, text="Enable", variable=self.crop_enabled_var,
                        command=self.render_post_frame).grid(row=0, column=0, padx=4)
        ttk.Label(crop_controls, text="Diameter").grid(row=0, column=1)
        self.crop_diameter_var = tk.DoubleVar(value=2)
        self.crop_slider = ttk.Scale(
            crop_controls, from_=2, to=1000, variable=self.crop_diameter_var,
            command=self.on_crop_diameter,
        )
        self.crop_slider.grid(row=0, column=2, sticky="ew", padx=8)
        self.crop_size_label = ttk.Label(crop_controls, text="2 px", width=9)
        self.crop_size_label.grid(row=0, column=3)
        ttk.Button(crop_controls, text="Fit to workspace", command=self.fit_crop_workspace
                   ).grid(row=0, column=4, padx=4)
        self.crop_export_button = ttk.Button(
            crop_controls, text="Export cropped video", command=self.start_crop_export)
        self.crop_export_button.grid(row=0, column=5, padx=4)
        self.crop_cancel_button = ttk.Button(
            crop_controls, text="Cancel export", state="disabled",
            command=self.crop_export_stop.set)
        self.crop_cancel_button.grid(row=1, column=5, padx=4)
        self.crop_status = ttk.Label(
            crop_controls, wraplength=520,
            text="Drag inside to move; drag the edge to resize. Export has black corners and no audio.")
        self.crop_status.grid(row=1, column=0, columnspan=5, sticky="w")

        overlay_controls = ttk.LabelFrame(
            post_processing, text="Transparent annotation overlay", padding=6
        )
        overlay_controls.grid(row=5, column=0, columnspan=6, sticky="ew", pady=(6, 0))
        self.overlay_chroma_button = ttk.Button(
            overlay_controls,
            text="Export blue-screen MP4",
            command=self.start_chroma_export,
        )
        self.overlay_chroma_button.grid(row=0, column=0, padx=4)
        self.overlay_export_button = ttk.Button(
            overlay_controls,
            text="Export transparent AVI",
            command=self.start_annotation_export,
        )
        self.overlay_export_button.grid(row=0, column=1, padx=4)
        self.overlay_sequence_button = ttk.Button(
            overlay_controls,
            text="Export PNG sequence",
            command=self.start_annotation_sequence_export,
        )
        self.overlay_sequence_button.grid(row=0, column=2, padx=4)
        self.overlay_cancel_button = ttk.Button(
            overlay_controls, text="Cancel export", state="disabled",
            command=self.overlay_export_stop.set,
        )
        self.overlay_cancel_button.grid(row=0, column=3, padx=4)
        self.overlay_export_status = ttk.Label(
            overlay_controls,
            text=(
                "MP4 uses a pure blue background (#0000FF) for Camtasia Remove a Color."
            ),
        )
        self.overlay_export_status.grid(row=1, column=0, columnspan=6, sticky="w")

        ttk.Label(post_processing, text="Expected robots:").grid(row=6, column=0, sticky="w")
        ttk.Entry(post_processing, textvariable=self.robot_count_var, width=6
                  ).grid(row=6, column=1, sticky="w")
        ttk.Label(post_processing, text="Center: equal weights · count mismatch = gap · no smoothing"
                  ).grid(row=6, column=2, columnspan=4, sticky="w")
        self.robot_count_var.trace_add("write", self.on_center_count_changed)

        self.status_label = tk.Label(
            self.root,
            text="Choose a recorded video, configure detection, and press Process Video.",
            fg="blue",
            anchor="w",
            padx=10,
        )
        self.status_label.pack(fill="x")
        self.detection_label = tk.Label(
            self.root, text="Detected robots: 0", fg="purple", anchor="w", padx=10
        )
        self.detection_label.pack(fill="x")

    def build_viewer(self):
        self.canvas = tk.Canvas(
            self.root, background="black", highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True, padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_crop_drag)
        self.canvas.bind("<ButtonRelease-1>", self.end_crop_drag)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

    def choose_video(self):
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        selected = filedialog.askopenfilename(
            title="Select experimental video",
            initialdir=INPUT_DIR,
            filetypes=VIDEO_FILETYPES,
        )
        if selected:
            self.video_path_var.set(selected)
            self.open_calibration_video(Path(selected))

    def open_calibration_video(self, video_path):
        self.pause_playback()
        self.pause_post_playback()
        if self.preview_capture is not None:
            self.preview_capture.release()

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            messagebox.showerror("Video Error", f"Could not open video: {video_path}")
            self.preview_capture = None
            return

        self.preview_capture = capture
        self.current_video_path = video_path
        self.video_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if self.video_fps <= 0:
            self.video_fps = 30.0
        self.video_total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.crop_enabled_var.set(False)
        self.crop_center = (width // 2, height // 2)
        self.detection_circle_enabled.set(False)
        self.detection_circle_edit.set(False)
        self.detection_slider.configure(to=min(width, height))
        self.set_detection_circle((width // 2, height // 2), min(width, height) // 2)
        self.crop_slider.configure(to=max(2, min(width, height)))
        self.set_crop(self.crop_center, min(width, height) // 2)
        self.timeline.config(to=max(self.video_total_frames - 1, 1))
        self.calibration_points = []
        self.robot_points = []
        self.current_cargo_detection = None
        self.learned_robot_color_ranges = None
        self.post_detections = {}
        self.post_center_track = []
        self.post_cargo_detections = {}
        self.post_cargo_track = []
        self.post_frame_number = 0
        self.post_timeline.config(to=max(self.video_total_frames - 1, 1))
        self.load_saved_calibration(video_path, width, height)
        self.show_calibration_frame(0)
        self.phase_notebook.select(0)

    def choose_post_detections(self):
        video_path = Path(self.video_path_var.get().strip())
        if not video_path.is_file():
            messagebox.showerror("Video Error", "Select the original video first.")
            return
        output_dir = PROJECT_DIR / "outputs" / "offline_detection" / video_path.stem
        selected = filedialog.askopenfilename(
            title="Select detection CSV", initialdir=output_dir,
            filetypes=[("Detection CSV", "*.csv")],
        )
        if selected:
            self.load_post_detections(Path(selected))

    def load_post_detections(self, csv_path):
        video_path = Path(self.video_path_var.get().strip())
        if not video_path.is_file():
            messagebox.showerror("Video Error", "Select the original video first.")
            return
        try:
            detections = read_detections_csv(csv_path)
            cargo_path = Path(csv_path).with_name(
                Path(csv_path).name.replace("_detections.csv", "_cargo.csv")
            )
            cargo_file_exists = cargo_path.is_file()
            cargo_detections = read_cargo_csv(cargo_path)
        except (OSError, ValueError, KeyError) as exc:
            messagebox.showerror("Detection Error", f"Could not read detections: {exc}")
            return
        if self.preview_capture is None or self.current_video_path != video_path:
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                messagebox.showerror("Video Error", f"Could not open video: {video_path}")
                return
            if self.preview_capture is not None:
                self.preview_capture.release()
            self.preview_capture = capture
            self.current_video_path = video_path
            self.video_fps = float(capture.get(cv2.CAP_PROP_FPS))
            if self.video_fps <= 0:
                self.video_fps = 30.0
            self.video_total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.timeline.config(to=max(self.video_total_frames - 1, 1))
        self.pause_post_playback()
        self.post_detections = detections
        self.post_cargo_detections = cargo_detections
        self.post_center_track = [
            (frame, robot_center_for(detections[frame], self.expected_center_count()))
            for frame in sorted(detections)
        ]
        self.post_cargo_track = [
            (frame, cargo_detections[frame]["center"])
            for frame in sorted(cargo_detections)
        ]
        if cargo_detections:
            self.post_cargo_status_label.config(
                text=(
                    f"Cargo data: loaded {len(cargo_detections)} detected frames "
                    f"from {cargo_path.name}."
                )
            )
        elif cargo_file_exists:
            self.post_cargo_status_label.config(
                text=(
                    "Cargo data: the cargo CSV exists, but no red cargo was detected. "
                    "Lower Cargo min area or check the circular detection area, then reprocess."
                )
            )
        else:
            self.post_cargo_status_label.config(
                text=(
                    "Cargo data: no companion cargo CSV. Return to Phase 2, tick "
                    "Detect red cargo, verify Preview cargo, then Process Video again."
                )
            )
        self.post_frame_number = 0
        self.post_timeline.config(to=max(self.video_total_frames - 1, 1))
        self.phase_notebook.select(3)
        self.show_post_frame(0)

    def show_post_frame(self, frame_number, sequential=False):
        if self.preview_capture is None:
            return
        frame_number = max(0, min(int(frame_number), max(self.video_total_frames - 1, 0)))
        if not sequential:
            self.preview_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame_bgr = self.preview_capture.read()
        if not success:
            self.pause_post_playback()
            return
        self.current_frame_bgr = frame_bgr
        self.post_frame_number = frame_number
        self.post_timeline_is_updating = True
        self.post_timeline.set(frame_number)
        self.post_timeline_is_updating = False
        total_seconds = max(self.video_total_frames - 1, 0) / self.video_fps
        self.post_time_label.config(
            text=f"{self.format_time(frame_number / self.video_fps)} / "
                 f"{self.format_time(total_seconds)}"
        )
        self.render_post_frame()

    def render_post_frame(self):
        if self.current_frame_bgr is None or self.phase_notebook.index(
                self.phase_notebook.select()) != 3:
            return
        detections = self.post_detections.get(self.post_frame_number, [])
        source_frame = self.current_frame_bgr
        if self.show_center_track_var.get():
            source_frame = draw_dashed_center_track(
                source_frame, self.post_center_track, self.post_frame_number
            )
        if self.show_cargo_track_var.get() and self.post_cargo_track:
            source_frame = draw_dashed_center_track(
                source_frame, self.post_cargo_track, self.post_frame_number,
                color=(255, 255, 0),
            )
        annotated = draw_detection_overlays(
            source_frame, detections, self.calibration_points,
            self.show_robot_marks_var.get(), self.show_lines_var.get(),
            "rectangle" if self.post_robot_marker_style_var.get()
            == "Hollow rectangle" else "cross",
            self.expected_center_count(),
        )
        if self.show_cargo_var.get():
            annotated = draw_cargo_overlay(
                annotated,
                self.post_cargo_detections.get(self.post_frame_number),
            )
        if self.crop_enabled_var.get() and self.crop_center is not None:
            annotated = circular_video_frame(
                annotated, self.crop_center, self.crop_radius, preview=True)
        self.show_preview(Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)))
        expected = self.expected_center_count()
        valid = robot_center_for(detections, expected) is not None
        self.detection_label.config(
            text=f"Detected robots: {len(detections)} / {expected if expected > 0 else '?'}"
                 + (" — center valid" if valid else " — center unavailable (check detections/count)"))

    def expected_center_count(self):
        try:
            return max(0, int(self.robot_count_var.get()))
        except ValueError:
            return 0

    def on_center_count_changed(self, *_args):
        expected = self.expected_center_count()
        self.post_center_track = [
            (frame, robot_center_for(items, expected))
            for frame, items in sorted(self.post_detections.items())
        ]
        if hasattr(self, "canvas"):
            self.render_post_frame()

    def detection_mask(self):
        enabled = getattr(self, "detection_circle_enabled", None)
        if enabled is None or not enabled.get() or self.detection_circle is None:
            return None
        return circle_detection_mask(self.current_frame_bgr.shape[:2], self.detection_circle)

    def set_detection_circle(self, center, radius):
        width = int(self.preview_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.preview_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        radius = max(1, min(int(round(radius)), min(width, height) // 2))
        center = (int(np.clip(round(center[0]), radius, width-radius)),
                  int(np.clip(round(center[1]), radius, height-radius)))
        self.detection_circle = (*center, radius)
        self.detection_diameter.set(2 * radius)
        self.detection_size_label.config(text=f"{2 * radius} px diameter")

    def resize_detection_circle(self, value):
        if self.detection_circle is not None:
            self.set_detection_circle(self.detection_circle[:2], float(value)/2)
            self.render_calibration_frame()

    def fit_detection_circle(self):
        if self.current_frame_bgr is None:
            return
        if len(self.calibration_points) == 4:
            center, radius = cv2.minEnclosingCircle(np.asarray(self.calibration_points, dtype=np.float32))
        else:
            height, width = self.current_frame_bgr.shape[:2]
            center, radius = (width/2, height/2), min(width, height)/2
        self.set_detection_circle(center, radius)
        self.detection_circle_enabled.set(True)
        # Fitting is a complete operation. Return immediately to robot-selection
        # mode instead of leaving all subsequent clicks assigned to circle edits.
        self.detection_circle_edit.set(False)
        self.robot_status_label.config(
            text="Circle fitted. Robot selection is active; click each robot.",
            fg="blue",
        )
        self.render_calibration_frame()

    def on_detection_edit_changed(self):
        if self.detection_circle_edit.get():
            self.robot_status_label.config(
                text=(
                    "Circle adjustment is active: drag the circle, then uncheck "
                    "Adjust circle to select robots."
                ),
                fg="orange",
            )
        else:
            self.robot_status_label.config(
                text="Robot selection is active; click each robot.",
                fg="blue",
            )
        self.render_calibration_frame()

    def set_crop(self, center, radius):
        if self.preview_capture is None:
            return
        width = int(self.preview_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.preview_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        radius = max(1, min(int(round(radius)), min(width, height) // 2))
        self.crop_center = (
            int(np.clip(round(center[0]), radius, width - radius)),
            int(np.clip(round(center[1]), radius, height - radius)),
        )
        self.crop_radius = radius
        self.crop_diameter_var.set(2 * radius)
        self.crop_size_label.config(text=f"{2 * radius} px")

    def on_crop_diameter(self, value):
        if self.crop_center is not None:
            self.set_crop(self.crop_center, float(value) / 2)
            self.render_post_frame()

    def fit_crop_workspace(self):
        if self.current_frame_bgr is None:
            return
        self.pause_post_playback()
        if len(self.calibration_points) == 4:
            center, radius = cv2.minEnclosingCircle(
                np.asarray(self.calibration_points, dtype=np.float32))
        else:
            height, width = self.current_frame_bgr.shape[:2]
            center, radius = (width / 2, height / 2), min(width, height) / 2
        self.set_crop(center, radius)
        self.crop_enabled_var.set(True)
        self.render_post_frame()

    def on_crop_drag(self, event):
        if self.detection_drag is not None and self.display_transform is not None:
            ox, oy, scale, _, _ = self.display_transform
            point = np.array([(event.x-ox)/scale, (event.y-oy)/scale])
            mode, anchor, center, radius = self.detection_drag
            self.set_detection_circle(center + point-anchor if mode == "move" else center,
                                      radius if mode == "move" else np.linalg.norm(point-center))
            self.render_calibration_frame()
            return
        if self.crop_drag is None or self.display_transform is None:
            return
        offset_x, offset_y, scale, _, _ = self.display_transform
        point = np.array([(event.x - offset_x) / scale, (event.y - offset_y) / scale])
        mode, anchor, center, radius = self.crop_drag
        if mode == "move":
            self.set_crop(center + point - anchor, radius)
        else:
            self.set_crop(center, np.linalg.norm(point - center))
        self.render_post_frame()

    def end_crop_drag(self, _event=None):
        self.crop_drag = None
        self.detection_drag = None

    def start_crop_export(self):
        if self.crop_export_thread is not None and self.crop_export_thread.is_alive():
            return
        if self.current_video_path is None or not self.crop_enabled_var.get():
            messagebox.showerror("Circular crop", "Open a video and enable circular crop first.")
            return
        destination = filedialog.asksaveasfilename(
            title="Export circular crop", defaultextension=".mp4",
            initialfile=f"{self.current_video_path.stem}_circular.mp4",
            filetypes=[("MP4 video", "*.mp4")],
        )
        if not destination:
            return
        if Path(destination).resolve() == Path(self.current_video_path).resolve():
            messagebox.showerror("Circular crop", "Choose a different file from the original video.")
            return
        self.pause_post_playback()
        self.crop_export_stop.clear()
        self.crop_export_button.config(state="disabled")
        self.crop_cancel_button.config(state="normal")
        self.crop_status.config(text="Exporting with the current circle and overlay settings…")
        self.crop_export_thread = threading.Thread(
            target=export_circular_video,
            args=(self.current_video_path, destination, tuple(self.crop_center),
                  self.crop_radius, dict(self.post_detections), list(self.post_center_track),
                  list(self.calibration_points), self.show_robot_marks_var.get(),
                  self.show_lines_var.get(), self.show_center_track_var.get(),
                  "rectangle" if self.post_robot_marker_style_var.get()
                  == "Hollow rectangle" else "cross",
                  self.crop_export_stop, self.crop_export_updates,
                  self.expected_center_count(), dict(self.post_cargo_detections),
                  list(self.post_cargo_track), self.show_cargo_var.get(),
                  self.show_cargo_track_var.get()),
            daemon=True,
        )
        self.crop_export_thread.start()
        self.root.after(100, self.poll_crop_export)

    def poll_crop_export(self):
        finished = False
        while not self.crop_export_updates.empty():
            item = self.crop_export_updates.get_nowait()
            if item[0] == "progress":
                self.crop_status.config(text=f"Exporting frame {item[1]} / {item[2] or '?'}…")
            else:
                finished = True
                text = (f"Saved {item[2]} frames: {item[1]}" if item[0] == "done"
                        else "Export cancelled." if item[0] == "cancelled"
                        else f"Export failed: {item[1]}")
                self.crop_status.config(text=text)
                self.crop_export_button.config(state="normal")
                self.crop_cancel_button.config(state="disabled")
        if not finished:
            self.root.after(100, self.poll_crop_export)

    def start_chroma_export(self):
        if (self.overlay_export_thread is not None
                and self.overlay_export_thread.is_alive()):
            return
        if self.current_video_path is None:
            messagebox.showerror(
                "Annotation overlay", "Open the original video first."
            )
            return
        destination = filedialog.asksaveasfilename(
            title="Export blue-screen annotation MP4",
            defaultextension=".mp4",
            initialfile=f"{self.current_video_path.stem}_annotations_blue.mp4",
            filetypes=[("MP4 video", "*.mp4")],
        )
        if not destination:
            return
        if Path(destination).resolve() == Path(self.current_video_path).resolve():
            messagebox.showerror(
                "Annotation overlay", "Choose a different file from the original video."
            )
            return
        self.pause_post_playback()
        self.overlay_export_stop.clear()
        self.overlay_chroma_button.config(state="disabled")
        self.overlay_export_button.config(state="disabled")
        self.overlay_sequence_button.config(state="disabled")
        self.overlay_cancel_button.config(state="normal")
        self.overlay_export_status.config(
            text="Exporting H.264 annotations over pure blue (#0000FF)…"
        )
        self.overlay_export_thread = threading.Thread(
            target=export_chroma_annotation_video,
            args=(
                self.current_video_path,
                destination,
                dict(self.post_detections),
                list(self.post_center_track),
                list(self.calibration_points),
                self.show_robot_marks_var.get(),
                self.show_lines_var.get(),
                self.show_center_track_var.get(),
                "rectangle" if self.post_robot_marker_style_var.get()
                == "Hollow rectangle" else "cross",
                self.expected_center_count(),
                dict(self.post_cargo_detections),
                list(self.post_cargo_track),
                self.show_cargo_var.get(),
                self.show_cargo_track_var.get(),
                self.overlay_export_stop,
                self.overlay_export_updates,
            ),
            daemon=True,
        )
        self.overlay_export_thread.start()
        self.root.after(100, self.poll_annotation_export)

    def start_annotation_export(self):
        if (self.overlay_export_thread is not None
                and self.overlay_export_thread.is_alive()):
            return
        if self.current_video_path is None:
            messagebox.showerror(
                "Annotation overlay", "Open the original video first."
            )
            return
        destination = filedialog.asksaveasfilename(
            title="Export transparent annotation overlay",
            defaultextension=".avi",
            initialfile=f"{self.current_video_path.stem}_annotations.avi",
            filetypes=[
                ("Transparent PNG video", "*.avi"),
                ("Transparent QuickTime video", "*.mov"),
            ],
        )
        if not destination:
            return
        if Path(destination).resolve() == Path(self.current_video_path).resolve():
            messagebox.showerror(
                "Annotation overlay", "Choose a different file from the original video."
            )
            return
        self.pause_post_playback()
        self.overlay_export_stop.clear()
        self.overlay_chroma_button.config(state="disabled")
        self.overlay_export_button.config(state="disabled")
        self.overlay_sequence_button.config(state="disabled")
        self.overlay_cancel_button.config(state="normal")
        self.overlay_export_status.config(
            text="Exporting lossless transparent annotations…"
        )
        self.overlay_export_thread = threading.Thread(
            target=export_annotation_overlay,
            args=(
                self.current_video_path,
                destination,
                dict(self.post_detections),
                list(self.post_center_track),
                list(self.calibration_points),
                self.show_robot_marks_var.get(),
                self.show_lines_var.get(),
                self.show_center_track_var.get(),
                "rectangle" if self.post_robot_marker_style_var.get()
                == "Hollow rectangle" else "cross",
                self.expected_center_count(),
                dict(self.post_cargo_detections),
                list(self.post_cargo_track),
                self.show_cargo_var.get(),
                self.show_cargo_track_var.get(),
                self.overlay_export_stop,
                self.overlay_export_updates,
            ),
            daemon=True,
        )
        self.overlay_export_thread.start()
        self.root.after(100, self.poll_annotation_export)

    def start_annotation_sequence_export(self):
        if (self.overlay_export_thread is not None
                and self.overlay_export_thread.is_alive()):
            return
        if self.current_video_path is None:
            messagebox.showerror(
                "Annotation overlay", "Open the original video first."
            )
            return
        parent = filedialog.askdirectory(
            title="Choose a folder for the transparent PNG sequence"
        )
        if not parent:
            return
        destination = Path(parent) / f"{self.current_video_path.stem}_annotations_png"
        if destination.exists() and any(destination.iterdir()):
            messagebox.showerror(
                "Annotation overlay",
                f"The output folder is not empty: {destination}",
            )
            return
        self.pause_post_playback()
        self.overlay_export_stop.clear()
        self.overlay_chroma_button.config(state="disabled")
        self.overlay_export_button.config(state="disabled")
        self.overlay_sequence_button.config(state="disabled")
        self.overlay_cancel_button.config(state="normal")
        self.overlay_export_status.config(
            text="Exporting transparent PNG image sequence…"
        )
        self.overlay_export_thread = threading.Thread(
            target=export_annotation_png_sequence,
            args=(
                self.current_video_path,
                destination,
                dict(self.post_detections),
                list(self.post_center_track),
                list(self.calibration_points),
                self.show_robot_marks_var.get(),
                self.show_lines_var.get(),
                self.show_center_track_var.get(),
                "rectangle" if self.post_robot_marker_style_var.get()
                == "Hollow rectangle" else "cross",
                self.expected_center_count(),
                dict(self.post_cargo_detections),
                list(self.post_cargo_track),
                self.show_cargo_var.get(),
                self.show_cargo_track_var.get(),
                self.overlay_export_stop,
                self.overlay_export_updates,
            ),
            daemon=True,
        )
        self.overlay_export_thread.start()
        self.root.after(100, self.poll_annotation_export)

    def poll_annotation_export(self):
        finished = False
        while not self.overlay_export_updates.empty():
            item = self.overlay_export_updates.get_nowait()
            if item[0] == "progress":
                self.overlay_export_status.config(
                    text=f"Exporting frame {item[1]} / {item[2] or '?'}…"
                )
            else:
                finished = True
                text = (
                    f"Saved {item[2]} transparent frames: {item[1]}"
                    if item[0] == "done"
                    else "Transparent-overlay export cancelled."
                    if item[0] == "cancelled"
                    else f"Transparent-overlay export failed: {item[1]}"
                )
                self.overlay_export_status.config(text=text)
                self.overlay_chroma_button.config(state="normal")
                self.overlay_export_button.config(state="normal")
                self.overlay_sequence_button.config(state="normal")
                self.overlay_cancel_button.config(state="disabled")
        if not finished:
            self.root.after(100, self.poll_annotation_export)

    def on_post_timeline_changed(self, value):
        if self.post_timeline_is_updating or self.preview_capture is None:
            return
        self.pause_post_playback()
        self.show_post_frame(round(float(value)))

    def toggle_post_playback(self):
        if self.preview_capture is None:
            messagebox.showerror("Video Error", "Select the original video first.")
            return
        if self.post_playing:
            self.pause_post_playback()
            return
        self.pause_playback()
        if self.post_frame_number >= self.video_total_frames - 1:
            self.show_post_frame(0)
        self.post_playing = True
        self.post_play_button.config(text="Pause")
        self.play_next_post_frame()

    def play_next_post_frame(self):
        if not self.post_playing:
            return
        next_frame = self.post_frame_number + 1
        if next_frame >= self.video_total_frames:
            self.pause_post_playback()
            return
        self.show_post_frame(next_frame, sequential=True)
        self.post_after_id = self.root.after(
            max(1, round(1000.0 / self.video_fps)), self.play_next_post_frame
        )

    def pause_post_playback(self):
        self.post_playing = False
        if hasattr(self, "post_play_button"):
            self.post_play_button.config(text="Play")
        if self.post_after_id is not None:
            self.root.after_cancel(self.post_after_id)
            self.post_after_id = None

    def load_saved_calibration(self, video_path, video_width, video_height):
        calibration_file = calibration_file_for(video_path)
        if not calibration_file.is_file():
            return
        try:
            data = json.loads(calibration_file.read_text(encoding="utf-8"))
            saved_size = data.get("video_size_px", [])
            points = data.get("points_px", [])
            if saved_size != [video_width, video_height] or len(points) != 4:
                return
            self.calibration_points = [tuple(map(int, point)) for point in points]
            self.object_height_var.set(str(data.get("h_cm", 1)))
            self.camera_height_var.set(str(data.get("camera_height_cm", 29)))
            self.rectangle_width_var.set(str(data.get("rectangle_width_cm", 10)))
            self.rectangle_height_var.set(str(data.get("rectangle_height_cm", 10)))
            self.calibration_minimum_area_var.set(
                str(data.get("calibration_minimum_area_px", 1))
            )
            self.calibration_status_label.config(
                text="Loaded the saved 4-point calibration for this video size.",
                fg="green",
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            self.calibration_points = []

    def show_calibration_frame(self, frame_number, sequential=False):
        if self.preview_capture is None:
            return
        frame_number = max(0, min(int(frame_number), max(self.video_total_frames - 1, 0)))
        if not sequential:
            self.preview_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame_bgr = self.preview_capture.read()
        if not success:
            self.pause_playback()
            return

        self.current_frame_number = frame_number
        self.current_frame_bgr = frame_bgr
        self.timeline_is_updating = True
        self.timeline.set(frame_number)
        self.timeline_is_updating = False
        current_seconds = frame_number / self.video_fps
        total_seconds = (
            max(self.video_total_frames - 1, 0) / self.video_fps
            if self.video_total_frames
            else 0
        )
        self.time_label.config(
            text=f"{self.format_time(current_seconds)} / {self.format_time(total_seconds)}"
        )
        self.render_calibration_frame()

    @staticmethod
    def format_time(seconds):
        minutes, seconds = divmod(max(float(seconds), 0.0), 60.0)
        return f"{int(minutes):02d}:{seconds:06.3f}"

    def on_timeline_changed(self, value):
        if self.timeline_is_updating or self.preview_capture is None:
            return
        self.pause_playback()
        self.show_calibration_frame(round(float(value)))

    def toggle_playback(self):
        if self.preview_capture is None:
            video_path = Path(self.video_path_var.get().strip())
            if not video_path.is_file():
                messagebox.showerror("Video Error", "Select an existing video file.")
                return
            self.open_calibration_video(video_path)
        if self.video_playing:
            self.pause_playback()
            return
        if self.current_frame_number >= self.video_total_frames - 1:
            self.show_calibration_frame(0)
        self.video_playing = True
        self.play_button.config(text="Pause")
        self.play_next_frame()

    def play_next_frame(self):
        if not self.video_playing:
            return
        next_frame = self.current_frame_number + 1
        if next_frame >= self.video_total_frames:
            self.pause_playback()
            return
        self.show_calibration_frame(next_frame, sequential=True)
        delay_ms = max(1, round(1000.0 / self.video_fps))
        self.playback_after_id = self.root.after(delay_ms, self.play_next_frame)

    def pause_playback(self):
        self.video_playing = False
        if hasattr(self, "play_button"):
            self.play_button.config(text="Play")
        if self.playback_after_id is not None:
            self.root.after_cancel(self.playback_after_id)
            self.playback_after_id = None

    def on_canvas_click(self, event):
        phase = self.phase_notebook.index(self.phase_notebook.select())
        point = self.canvas_point_to_source(event.x, event.y)
        if point is None:
            return
        if phase == 0:
            self.add_calibration_point(point)
        elif phase == 1:
            if self.detection_circle_enabled.get() and self.detection_circle_edit.get():
                x, y, radius = self.detection_circle
                center = np.array([x, y], dtype=float)
                distance = np.linalg.norm(np.asarray(point)-center)
                tolerance = 10 / max(self.display_transform[2], 1e-6)
                if abs(distance-radius) <= tolerance or distance < radius:
                    self.pause_playback()
                    self.detection_drag = ("resize" if abs(distance-radius) <= tolerance else "move",
                                           np.asarray(point), center, radius)
            else:
                self.add_robot_point(point)
        elif phase == 3 and self.crop_enabled_var.get() and self.crop_center is not None:
            center = np.asarray(self.crop_center, dtype=float)
            distance = np.linalg.norm(np.asarray(point) - center)
            tolerance = 10 / max(self.display_transform[2], 1e-6)
            if abs(distance - self.crop_radius) <= tolerance:
                mode = "resize"
            elif distance < self.crop_radius:
                mode = "move"
            else:
                return
            self.pause_post_playback()
            self.crop_drag = (mode, np.asarray(point), center, self.crop_radius)

    def canvas_point_to_source(self, canvas_x, canvas_y):
        if self.current_frame_bgr is None or self.display_transform is None:
            return None
        offset_x, offset_y, scale, source_width, source_height = self.display_transform
        source_x = round((canvas_x - offset_x) / scale)
        source_y = round((canvas_y - offset_y) / scale)
        if not (0 <= source_x < source_width and 0 <= source_y < source_height):
            return None
        return source_x, source_y

    def add_calibration_point(self, point):
        if len(self.calibration_points) >= 4:
            self.calibration_status_label.config(
                text="Four points are already selected. Clear them to select again.",
                fg="orange",
            )
            return

        snap_result = self.snap_calibration_point(point)
        if snap_result is None:
            return
        snapped_point, snapped, distance = snap_result
        self.calibration_points.append(snapped_point)
        if snapped:
            detail = f" Snapped {distance:.1f} px to the green-dot center."
            status_color = "blue"
        else:
            detail = " No nearby green dot found; kept the clicked position."
            status_color = "orange"
        if len(self.calibration_points) == 4:
            self.calibration_points = self.order_corner_points(self.calibration_points)
            message = "4/4 dots selected. Check the green quadrilateral, then save."
            if snapped:
                status_color = "green"
        else:
            message = f"{len(self.calibration_points)}/4 dots selected."
        self.calibration_status_label.config(
            text=message + detail, fg=status_color
        )
        self.render_calibration_frame()

    def snap_calibration_point(self, clicked_point):
        """Snap an approximate click to the nearest green calibration dot."""
        if self.current_frame_bgr is None:
            return clicked_point, False, 0.0
        try:
            radius = int(self.calibration_search_radius_var.get().strip())
            minimum_area = float(self.calibration_minimum_area_var.get().strip())
            if radius <= 0 or minimum_area <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Input Error", "Green-dot search radius and min area must be positive."
            )
            return None
        return self.find_nearest_blob_center(
            clicked_point,
            color="Green",
            radius=radius,
            minimum_area=minimum_area,
            morphology_kernel_size=1,
        )

    def add_robot_point(self, point):
        mask = self.detection_mask()
        if mask is not None and not mask[point[1], point[0]]:
            self.robot_status_label.config(text="Select a robot inside the circular detection area.", fg="orange")
            return
        if self.robot_finding_mode_var.get() != "Manual selection":
            self.robot_status_label.config(
                text="Choose Manual selection to mark robots with the mouse.",
                fg="orange",
            )
            return
        robot_count = self.read_robot_count()
        if robot_count is None:
            return
        exclusion_radius = self.read_calibration_exclusion_radius()
        if exclusion_radius is None:
            return
        if exclusion_radius > 0 and self.calibration_points and min(
            np.hypot(point[0] - marker[0], point[1] - marker[1])
            for marker in self.calibration_points
        ) <= exclusion_radius:
            self.robot_status_label.config(
                text=(
                    "That click is inside a fixed calibration-dot exclusion area; "
                    "click the robot instead."
                ),
                fg="orange",
            )
            return
        if len(self.robot_points) >= robot_count:
            self.robot_status_label.config(
                text=f"All {robot_count} robots are selected. Clear them to select again.",
                fg="orange",
            )
            return
        snap_result = self.snap_robot_point(point)
        if snap_result is None:
            return
        snapped_point, snapped, distance = snap_result
        self.robot_points.append(snapped_point)
        learned_detail = ""
        if self.detection_color_var.get() == "Learned from clicks":
            learned_detail = " " + self.update_learned_robot_color_profile()
        selected_count = len(self.robot_points)
        if snapped:
            detail = f"R{selected_count} snapped {distance:.1f} px to the blob center."
            status_color = "green" if selected_count == robot_count else "blue"
        else:
            detail = (
                f"R{selected_count}: no matching blob nearby; kept the clicked position."
            )
            status_color = "orange"
        self.robot_status_label.config(
            text=f"{selected_count}/{robot_count} robots selected. {detail}{learned_detail}",
            fg=status_color,
        )
        self.render_calibration_frame()

    def snap_robot_point(self, clicked_point):
        """Snap an approximate click to the closest selected-color blob center."""
        if self.current_frame_bgr is None:
            return clicked_point, False, 0.0
        try:
            radius = int(self.robot_search_radius_var.get().strip())
            if radius <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Input Error", "Search radius and min area must be positive."
            )
            return None

        if self.detection_color_var.get() == "Learned from clicks":
            return self.find_local_contrasting_point(clicked_point, radius)

        try:
            minimum_area = float(self.robot_snap_minimum_area_var.get().strip())
            if minimum_area <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Input Error", "Click min area must be positive.")
            return None

        return self.find_nearest_blob_center(
            clicked_point,
            color=self.detection_color_var.get(),
            radius=radius,
            minimum_area=minimum_area,
            morphology_kernel_size=1,
            detection_mask=self.detection_mask(),
        )

    def find_local_contrasting_point(self, clicked_point, radius):
        """Find the darkest, most saturated pixel close to a learning click."""
        clicked_x, clicked_y = clicked_point
        frame_height, frame_width = self.current_frame_bgr.shape[:2]
        local_radius = min(radius, 10)
        x0 = max(0, clicked_x - local_radius)
        y0 = max(0, clicked_y - local_radius)
        x1 = min(frame_width, clicked_x + local_radius + 1)
        y1 = min(frame_height, clicked_y + local_radius + 1)
        hsv = cv2.cvtColor(self.current_frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        score = hsv[:, :, 2].astype(np.float32) - 0.35 * hsv[:, :, 1]
        mask = self.detection_mask()
        if mask is not None:
            score[mask[y0:y1, x0:x1] == 0] = np.inf
            if not np.isfinite(score).any():
                return clicked_point, False, 0.0
        local_y, local_x = np.unravel_index(np.argmin(score), score.shape)
        if float(np.median(score[np.isfinite(score)]) - score[local_y, local_x]) < 8.0:
            return clicked_point, False, 0.0
        center = (int(local_x + x0), int(local_y + y0))
        distance = float(np.hypot(center[0] - clicked_x, center[1] - clicked_y))
        return center, True, distance

    def update_learned_robot_color_profile(self):
        """Build an HSV profile from small neighborhoods around clicked robots."""
        hsv_frame = cv2.cvtColor(self.current_frame_bgr, cv2.COLOR_BGR2HSV)
        samples = []
        height, width = hsv_frame.shape[:2]
        for center_x, center_y in self.robot_points:
            x0, x1 = max(0, center_x - 2), min(width, center_x + 3)
            y0, y1 = max(0, center_y - 2), min(height, center_y + 3)
            patch = hsv_frame[y0:y1, x0:x1].reshape(-1, 3)
            scores = patch[:, 2].astype(np.float32) - 0.35 * patch[:, 1]
            selected = patch[scores <= scores.min() + 20.0]
            samples.extend(selected.tolist())

        values = np.asarray(samples, dtype=np.int16)
        hue_angles = values[:, 0] * (2.0 * np.pi / 180.0)
        hue_center = int(round(
            np.arctan2(np.sin(hue_angles).mean(), np.cos(hue_angles).mean())
            * 180.0 / (2.0 * np.pi)
        )) % 180
        hue_offsets = (values[:, 0] - hue_center + 90) % 180 - 90
        hue_spread = min(25, int(np.max(np.abs(hue_offsets))) + 8)
        saturation_low = max(0, int(values[:, 1].min()) - 25)
        saturation_high = min(255, int(values[:, 1].max()) + 25)
        value_low = max(0, int(values[:, 2].min()) - 25)
        value_high = min(255, int(values[:, 2].max()) + 35)
        hue_low, hue_high = hue_center - hue_spread, hue_center + hue_spread
        if hue_low < 0:
            ranges = [
                ((0, saturation_low, value_low), (hue_high, saturation_high, value_high)),
                ((180 + hue_low, saturation_low, value_low), (179, saturation_high, value_high)),
            ]
        elif hue_high > 179:
            ranges = [
                ((hue_low, saturation_low, value_low), (179, saturation_high, value_high)),
                ((0, saturation_low, value_low), (hue_high - 180, saturation_high, value_high)),
            ]
        else:
            ranges = [
                ((hue_low, saturation_low, value_low), (hue_high, saturation_high, value_high))
            ]
        self.learned_robot_color_ranges = ranges
        self.minimum_area_var.set("1")
        self.robot_snap_minimum_area_var.set("1")
        self.detection_filter_size_var.set("1")
        return (
            f"Learned HSV near H={hue_center}, "
            f"S={saturation_low}-{saturation_high}, V={value_low}-{value_high}."
        )

    def selected_robot_color_ranges(self):
        if self.detection_color_var.get() != "Learned from clicks":
            return None
        if self.learned_robot_color_ranges:
            return self.learned_robot_color_ranges
        messagebox.showerror(
            "Robot Profile Error",
            "Choose Manual selection and click the robots to learn their appearance first.",
        )
        return False

    def find_nearest_blob_center(
        self,
        clicked_point,
        color,
        radius,
        minimum_area,
        morphology_kernel_size=5,
        detection_mask=None,
    ):
        clicked_x, clicked_y = clicked_point
        frame_height, frame_width = self.current_frame_bgr.shape[:2]
        x0 = max(0, clicked_x - radius)
        y0 = max(0, clicked_y - radius)
        x1 = min(frame_width, clicked_x + radius + 1)
        y1 = min(frame_height, clicked_y + radius + 1)
        search_area = self.current_frame_bgr[y0:y1, x0:x1].copy()
        _, detections = self.detector.process(
            search_area,
            mode="Color blobs",
            color=color,
            minimum_area=minimum_area,
            morphology_kernel_size=morphology_kernel_size,
            detection_mask=None if detection_mask is None else detection_mask[y0:y1, x0:x1],
        )

        candidates = []
        for detection in detections:
            local_x, local_y = detection["center"]
            center = (local_x + x0, local_y + y0)
            distance = float(np.hypot(center[0] - clicked_x, center[1] - clicked_y))
            if distance <= radius:
                candidates.append((distance, center))
        if not candidates:
            return clicked_point, False, 0.0
        distance, center = min(candidates, key=lambda candidate: candidate[0])
        return center, True, distance

    def read_robot_count(self):
        try:
            robot_count = int(self.robot_count_var.get().strip())
            if robot_count <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Input Error", "Number of robots must be positive.")
            return None
        return robot_count

    def read_detection_filter_size(self):
        try:
            filter_size = int(self.detection_filter_size_var.get().strip())
            if filter_size not in (1, 3, 5):
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Input Error", "Detection cleanup size must be 1, 3, or 5."
            )
            return None
        return filter_size

    def read_calibration_exclusion_radius(self):
        variable = getattr(self, "calibration_exclusion_radius_var", None)
        if variable is None:
            return 12.0
        try:
            radius = float(variable.get().strip())
            if radius < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Input Error", "Calibration-dot exclusion must be zero or positive."
            )
            return None
        return radius

    def read_cargo_settings(self):
        try:
            minimum_area = float(self.cargo_minimum_area_var.get().strip())
            tracking_radius = float(self.cargo_tracking_radius_var.get().strip())
            if minimum_area <= 0 or tracking_radius <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Cargo Input Error",
                "Cargo minimum area and tracking radius must be positive.",
            )
            return None
        return minimum_area, tracking_radius

    def detect_cargo_in_frame(self, frame_bgr, minimum_area):
        _, candidates = self.detector.process(
            frame_bgr,
            mode="Color blobs",
            color="Red",
            minimum_area=minimum_area,
            morphology_kernel_size=3,
            draw_annotations=False,
            detection_mask=self.detection_mask(),
        )
        return select_cargo_candidate(candidates)

    def on_cargo_toggle(self):
        if self.cargo_enabled_var.get():
            self.preview_cargo()
        else:
            self.current_cargo_detection = None
            self.cargo_status_label.config(
                text="Disabled; robot detection is unchanged."
            )
            self.render_calibration_frame()

    def preview_cargo(self):
        if not self.cargo_enabled_var.get():
            self.cargo_status_label.config(
                text="Tick Detect red cargo to enable cargo measurement."
            )
            return
        if self.current_frame_bgr is None:
            self.cargo_status_label.config(text="Choose a video first.")
            return
        settings = self.read_cargo_settings()
        if settings is None:
            return
        minimum_area, _ = settings
        self.current_cargo_detection = self.detect_cargo_in_frame(
            self.current_frame_bgr, minimum_area
        )
        if self.current_cargo_detection is None:
            self.cargo_status_label.config(
                text="No red cargo found. Check the circular area and minimum area."
            )
        else:
            center = self.current_cargo_detection["center"]
            self.cargo_status_label.config(
                text=f"Red cargo found at {center}; cyan box/center is the cargo measurement."
            )
        self.render_calibration_frame()

    def find_robots_by_color(self):
        if self.current_frame_bgr is None:
            messagebox.showerror("Video Error", "Choose a video first.")
            return
        if len(self.calibration_points) != 4:
            messagebox.showerror(
                "Calibration Error", "Select the four calibration dots first."
            )
            return
        robot_count = self.read_robot_count()
        if robot_count is None:
            return
        filter_size = self.read_detection_filter_size()
        if filter_size is None:
            return
        color_ranges = self.selected_robot_color_ranges()
        if color_ranges is False:
            return
        exclusion_radius = self.read_calibration_exclusion_radius()
        if exclusion_radius is None:
            return
        try:
            minimum_area = float(self.minimum_area_var.get().strip())
            if minimum_area <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Input Error", "Min area must be positive.")
            return

        _, detections = self.detector.process(
            self.current_frame_bgr.copy(),
            detection_mask=self.detection_mask(),
            mode="Color blobs",
            color=self.detection_color_var.get(),
            minimum_area=minimum_area,
            morphology_kernel_size=filter_size,
            color_ranges=color_ranges,
        )
        eligible, excluded_indices = exclude_calibration_markers(
            detections, self.calibration_points, exclusion_radius
        )
        previous_points = list(self.robot_points)
        try:
            tracking_radius = int(self.robot_search_radius_var.get().strip())
            if tracking_radius <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Input Error", "Click/tracking radius must be positive.")
            return

        if len(previous_points) == robot_count:
            selected = match_robot_candidates(
                eligible, previous_points, tracking_radius
            )
        else:
            selected = []
            for robot_id, detection in enumerate(eligible[:robot_count], start=1):
                accepted = dict(detection)
                accepted["robot_id"] = robot_id
                selected.append(accepted)
        self.robot_points = [item["center"] for item in selected]
        self.robot_finding_mode_var.set("Color detection")
        found = len(self.robot_points)
        self.robot_status_label.config(
            text=(
                f"Selected {found}/{robot_count} robots from "
                f"{len(eligible)} eligible color candidates; "
                f"{len(excluded_indices)} calibration-dot candidate(s) ignored. "
                "Verify the R labels."
            ),
            fg="green" if found == robot_count else "orange",
        )
        self.detection_label.config(
            text=(
                f"Selected robots: {found}/{robot_count} "
                f"({len(eligible)} eligible, {len(excluded_indices)} calibration ignored)"
            )
        )
        self.render_calibration_frame()

    def clear_robot_points(self):
        self.robot_points = []
        self.robot_status_label.config(
            text="Robot selections cleared. Click robots or find them by color.",
            fg="blue",
        )
        self.render_calibration_frame()

    @staticmethod
    def order_corner_points(points):
        """Return top-left, top-right, bottom-right, bottom-left corner order."""
        top_left = min(points, key=lambda point: point[0] + point[1])
        bottom_right = max(points, key=lambda point: point[0] + point[1])
        top_right = max(points, key=lambda point: point[0] - point[1])
        bottom_left = min(points, key=lambda point: point[0] - point[1])
        return [top_left, top_right, bottom_right, bottom_left]

    def render_calibration_frame(self):
        if self.current_frame_bgr is None:
            return
        annotated = self.current_frame_bgr.copy()
        for index, point in enumerate(self.calibration_points, start=1):
            cv2.circle(annotated, point, 8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(
                annotated,
                f"P{index}",
                (point[0] + 10, point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        if len(self.calibration_points) >= 2:
            points = np.array(self.calibration_points, dtype=np.int32)
            cv2.polylines(
                annotated,
                [points],
                len(self.calibration_points) == 4,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        if self.phase_notebook.index(self.phase_notebook.select()) == 1:
            for robot_id, point in enumerate(self.robot_points, start=1):
                x, y = point
                cv2.rectangle(
                    annotated, (x - 10, y - 10), (x + 10, y + 10),
                    (0, 255, 0), 2, cv2.LINE_AA,
                )
                cv2.putText(
                    annotated, f"R{robot_id}", (x + 12, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                    cv2.LINE_AA,
                )
            if self.cargo_enabled_var.get() and self.current_cargo_detection:
                annotated = draw_cargo_overlay(
                    annotated, self.current_cargo_detection
                )
            if self.detection_mask() is not None:
                annotated = circular_video_frame(annotated, self.detection_circle[:2],
                                                 self.detection_circle[2], preview=True)
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        self.show_preview(Image.fromarray(rgb))

    def clear_calibration_points(self):
        self.calibration_points = []
        self.calibration_status_label.config(
            text="Points cleared. Click the 4 green dots.", fg="blue"
        )
        self.render_calibration_frame()

    def save_calibration(self):
        if len(self.calibration_points) != 4 or self.current_frame_bgr is None:
            messagebox.showerror(
                "Calibration Error", "Select all four green dots before saving."
            )
            return
        geometry = self.read_geometry_values()
        if geometry is None:
            return
        try:
            calibration_minimum_area = float(
                self.calibration_minimum_area_var.get().strip()
            )
            if calibration_minimum_area <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Input Error", "Green-dot min area must be positive."
            )
            return
        object_height_cm, camera_height_cm, rectangle_width_cm, rectangle_height_cm = geometry
        height, width = self.current_frame_bgr.shape[:2]
        calibration = {
            "video_size_px": [width, height],
            "calibration_frame": self.current_frame_number,
            "points_px": [list(point) for point in self.calibration_points],
            "point_order": ["top_left", "top_right", "bottom_right", "bottom_left"],
            "h_cm": object_height_cm,
            "camera_height_cm": camera_height_cm,
            "rectangle_width_cm": rectangle_width_cm,
            "rectangle_height_cm": rectangle_height_cm,
            "calibration_minimum_area_px": calibration_minimum_area,
        }
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        video_path = self.current_video_path or Path(self.video_path_var.get().strip())
        calibration_file = calibration_file_for(video_path)
        calibration_file.write_text(
            json.dumps(calibration, indent=2) + "\n", encoding="utf-8"
        )
        self.calibration_status_label.config(
            text=f"Calibration saved: {calibration_file}", fg="green"
        )

    def read_geometry_values(self):
        try:
            object_height_cm = float(self.object_height_var.get().strip())
            camera_height_cm = float(self.camera_height_var.get().strip())
            rectangle_width_cm = float(self.rectangle_width_var.get().strip())
            rectangle_height_cm = float(self.rectangle_height_var.get().strip())
            if (
                object_height_cm < 0
                or camera_height_cm <= 0
                or rectangle_width_cm <= 0
                or rectangle_height_cm <= 0
            ):
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Input Error",
                "h must be zero or positive; camera height and rectangle sizes "
                "must be positive.",
            )
            return None
        return (
            object_height_cm,
            camera_height_cm,
            rectangle_width_cm,
            rectangle_height_cm,
        )

    def on_phase_changed(self, _event=None):
        self.pause_playback()
        self.pause_post_playback()
        phase = self.phase_notebook.index(self.phase_notebook.select())
        if phase in (0, 1):
            if self.preview_capture is not None:
                self.show_calibration_frame(self.current_frame_number)
            else:
                self.render_calibration_frame()
        elif phase == 3 and self.preview_capture is not None:
            self.show_post_frame(self.post_frame_number)

    def on_canvas_resize(self, _event=None):
        if not hasattr(self, "phase_notebook"):
            return
        phase = self.phase_notebook.index(self.phase_notebook.select())
        if phase in (0, 1):
            self.render_calibration_frame()
        elif phase == 3:
            self.render_post_frame()

    def start_processing(self):
        if self.worker_thread and self.worker_thread.is_alive():
            return

        video_path = Path(self.video_path_var.get().strip())
        if not video_path.is_file():
            messagebox.showerror("Video Error", "Select an existing video file.")
            return
        try:
            minimum_area = float(self.minimum_area_var.get().strip())
            if minimum_area <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Input Error", "Min area must be positive.")
            return
        geometry = self.read_geometry_values()
        if geometry is None:
            return
        (
            object_height_cm,
            camera_height_cm,
            rectangle_width_cm,
            rectangle_height_cm,
        ) = geometry
        if len(self.calibration_points) != 4:
            messagebox.showerror(
                "Calibration Error",
                "Complete Phase 1 by selecting the four green dots first.",
            )
            return
        robot_count = self.read_robot_count()
        if robot_count is None:
            return
        filter_size = self.read_detection_filter_size()
        if filter_size is None:
            return
        color_ranges = self.selected_robot_color_ranges()
        if color_ranges is False:
            return
        exclusion_radius = self.read_calibration_exclusion_radius()
        if exclusion_radius is None:
            return
        cargo_enabled = self.cargo_enabled_var.get()
        cargo_minimum_area, cargo_tracking_radius = (0.0, 0.0)
        if cargo_enabled:
            cargo_settings = self.read_cargo_settings()
            if cargo_settings is None:
                return
            cargo_minimum_area, cargo_tracking_radius = cargo_settings
        if len(self.robot_points) != robot_count:
            messagebox.showerror(
                "Robot Finding Error",
                f"Complete Phase 2 by selecting or finding all {robot_count} robots.",
            )
            return
        try:
            tracking_radius = int(self.robot_search_radius_var.get().strip())
            if tracking_radius <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Input Error", "Click/tracking radius must be positive.")
            return

        mode = "Color blobs"
        color = self.detection_color_var.get()
        self.pause_playback()
        self.stop_event = threading.Event()
        self.set_running(True)
        self.progress["value"] = 0
        self.status_label.config(text="Opening video...", fg="blue")

        self.worker_thread = threading.Thread(
            target=self.process_video,
            args=(
                video_path,
                mode,
                color,
                minimum_area,
                object_height_cm,
                camera_height_cm,
                rectangle_width_cm,
                rectangle_height_cm,
                list(self.calibration_points),
                self.stop_event,
                filter_size,
                color_ranges,
                robot_count,
                self.detection_circle if self.detection_circle_enabled.get() else None,
                list(self.robot_points),
                tracking_radius,
                exclusion_radius,
                cargo_enabled,
                cargo_minimum_area,
                cargo_tracking_radius,
            ),
            daemon=True,
        )
        self.worker_thread.start()

    def process_video(
        self,
        video_path,
        mode,
        color,
        minimum_area,
        object_height_cm,
        camera_height_cm,
        rectangle_width_cm,
        rectangle_height_cm,
        calibration_points,
        stop_event,
        morphology_kernel_size=5,
        color_ranges=None,
        expected_count=None,
        detection_circle=None,
        initial_robot_points=None,
        tracking_radius=35,
        calibration_exclusion_radius=0,
        cargo_enabled=False,
        cargo_minimum_area=200,
        cargo_tracking_radius=100,
    ):
        capture = cv2.VideoCapture(str(video_path))
        writer = None
        csv_file = None
        cargo_csv_file = None
        result = None
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Could not open video: {video_path}")

            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if fps <= 0:
                fps = 30.0
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if width <= 0 or height <= 0:
                raise RuntimeError("The video does not report a valid frame size.")
            detection_mask = circle_detection_mask((height, width), detection_circle)

            output_dir = PROJECT_DIR / "outputs" / "offline_detection" / video_path.stem
            output_dir.mkdir(parents=True, exist_ok=True)
            video_output = output_dir / f"{video_path.stem}_annotated.mp4"
            csv_output = output_dir / f"{video_path.stem}_detections.csv"
            cargo_output = output_dir / f"{video_path.stem}_cargo.csv"

            writer = cv2.VideoWriter(
                str(video_output),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not create output video: {video_output}")

            csv_file = csv_output.open("w", newline="", encoding="utf-8")
            csv_writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "frame", "time_s", "h_cm", "camera_height_cm",
                    "dot_rectangle_width_cm", "dot_rectangle_height_cm",
                    "detection", "kind", "label",
                    "center_x_px", "center_y_px", "area_px2", "angle_deg",
                    "candidate_count", "accepted", "robot_id",
                    "exclusion_reason",
                ],
            )
            csv_writer.writeheader()

            cargo_writer = None
            if cargo_enabled:
                cargo_csv_file = cargo_output.open("w", newline="", encoding="utf-8")
                cargo_writer = csv.DictWriter(
                    cargo_csv_file,
                    fieldnames=[
                        "frame", "time_s", "detected", "center_x_px", "center_y_px",
                        "area_px2", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
                    ],
                )
                cargo_writer.writeheader()

            frame_number = 0
            tracked_points = (
                [tuple(point) for point in initial_robot_points]
                if initial_robot_points else []
            )
            cargo_previous_center = None
            cargo_track = []
            while not stop_event.is_set():
                success, frame_bgr = capture.read()
                if not success:
                    break

                _, candidates = self.detector.process(
                    frame_bgr,
                    mode=mode,
                    color=color,
                    minimum_area=minimum_area,
                    morphology_kernel_size=morphology_kernel_size,
                    color_ranges=color_ranges,
                    draw_annotations=False,
                    detection_mask=detection_mask,
                )
                eligible, excluded_indices = exclude_calibration_markers(
                    candidates, calibration_points, calibration_exclusion_radius
                )
                cargo_detection = None
                if cargo_enabled:
                    _, cargo_candidates = self.detector.process(
                        frame_bgr,
                        mode="Color blobs",
                        color="Red",
                        minimum_area=cargo_minimum_area,
                        morphology_kernel_size=3,
                        draw_annotations=False,
                        detection_mask=detection_mask,
                    )
                    cargo_detection = select_cargo_candidate(
                        cargo_candidates,
                        cargo_previous_center,
                        cargo_tracking_radius,
                    )
                    if cargo_detection is not None:
                        cargo_previous_center = cargo_detection["center"]
                    cargo_track.append(
                        (frame_number, None if cargo_detection is None
                         else cargo_detection["center"])
                    )
                if expected_count is None:
                    selected = []
                    for robot_id, candidate in enumerate(eligible, start=1):
                        detection = dict(candidate)
                        detection["robot_id"] = robot_id
                        selected.append(detection)
                elif tracked_points:
                    selected = match_robot_candidates(
                        eligible, tracked_points, tracking_radius
                    )
                    for detection in selected:
                        tracked_points[detection["robot_id"] - 1] = detection["center"]
                else:
                    selected = []
                    for robot_id, candidate in enumerate(
                            eligible[:expected_count], start=1):
                        detection = dict(candidate)
                        detection["robot_id"] = robot_id
                        selected.append(detection)
                    tracked_points = [item["center"] for item in selected]
                annotated = draw_detection_overlays(
                    frame_bgr, selected, calibration_points,
                    expected_count=expected_count,
                )
                if cargo_enabled:
                    annotated = draw_dashed_center_track(
                        annotated, cargo_track, frame_number, color=(255, 255, 0)
                    )
                    annotated = draw_cargo_overlay(annotated, cargo_detection)
                writer.write(annotated)
                time_s = frame_number / fps

                if cargo_writer is not None:
                    bbox = cargo_detection.get("bbox", ("", "", "", "")) \
                        if cargo_detection else ("", "", "", "")
                    center = cargo_detection.get("center", ("", "")) \
                        if cargo_detection else ("", "")
                    cargo_writer.writerow({
                        "frame": frame_number,
                        "time_s": f"{time_s:.6f}",
                        "detected": 1 if cargo_detection else 0,
                        "center_x_px": center[0],
                        "center_y_px": center[1],
                        "area_px2": cargo_detection.get("area", "")
                        if cargo_detection else "",
                        "bbox_x": bbox[0], "bbox_y": bbox[1],
                        "bbox_w": bbox[2], "bbox_h": bbox[3],
                    })

                accepted_by_candidate = {
                    detection["candidate_index"]: detection
                    for detection in selected
                }
                if candidates:
                    for detection_number, detection in enumerate(candidates, start=1):
                        center = detection.get("center", (None, None))
                        accepted = accepted_by_candidate.get(detection_number - 1)
                        csv_writer.writerow({
                            "frame": frame_number,
                            "time_s": f"{time_s:.6f}",
                            "h_cm": object_height_cm,
                            "camera_height_cm": camera_height_cm,
                            "dot_rectangle_width_cm": rectangle_width_cm,
                            "dot_rectangle_height_cm": rectangle_height_cm,
                            "detection": detection_number,
                            "kind": detection.get("kind", ""),
                            "label": detection.get("label", ""),
                            "center_x_px": center[0],
                            "center_y_px": center[1],
                            "area_px2": detection.get("area", ""),
                            "angle_deg": detection.get("angle_deg", ""),
                            "candidate_count": len(candidates),
                            "accepted": 1 if accepted is not None else 0,
                            "robot_id": (accepted or {}).get("robot_id", ""),
                            "exclusion_reason": (
                                "calibration_marker"
                                if detection_number - 1 in excluded_indices else ""
                            ),
                        })
                else:
                    csv_writer.writerow({
                        "frame": frame_number,
                        "time_s": f"{time_s:.6f}",
                        "h_cm": object_height_cm,
                        "camera_height_cm": camera_height_cm,
                        "dot_rectangle_width_cm": rectangle_width_cm,
                        "dot_rectangle_height_cm": rectangle_height_cm,
                        "candidate_count": 0,
                    })

                expected_text = expected_count if expected_count is not None else len(selected)
                summary = (
                    f"Tracked robots: {len(selected)}/{expected_text} "
                    f"({len(eligible)} eligible, "
                    f"{len(excluded_indices)} calibration ignored)"
                )
                progress = (
                    100.0 * (frame_number + 1) / total_frames
                    if total_frames > 0
                    else 0.0
                )
                if frame_number % 5 == 0:
                    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    self.put_latest_preview({
                        "kind": "frame",
                        "image": Image.fromarray(rgb),
                        "summary": summary,
                        "frame": frame_number + 1,
                        "total": total_frames,
                        "progress": progress,
                    })
                frame_number += 1

            stopped = stop_event.is_set()
            result = {
                "kind": "finished",
                "stopped": stopped,
                "frames": frame_number,
                "video_output": video_output,
                "csv_output": csv_output,
                "cargo_output": cargo_output if cargo_enabled else None,
                "object_height_cm": object_height_cm,
                "camera_height_cm": camera_height_cm,
                "rectangle_width_cm": rectangle_width_cm,
                "rectangle_height_cm": rectangle_height_cm,
            }
        except Exception as exc:
            result = {"kind": "error", "message": str(exc)}
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            if csv_file is not None:
                csv_file.close()
            if cargo_csv_file is not None:
                cargo_csv_file.close()
        self.put_latest_preview(result)

    def put_latest_preview(self, item):
        try:
            self.preview_queue.put_nowait(item)
        except queue.Full:
            try:
                self.preview_queue.get_nowait()
            except queue.Empty:
                pass
            self.preview_queue.put_nowait(item)

    def poll_preview(self):
        item = None
        while not self.preview_queue.empty():
            try:
                item = self.preview_queue.get_nowait()
            except queue.Empty:
                break

        if item is not None:
            if item["kind"] == "frame":
                if self.phase_notebook.index(self.phase_notebook.select()) != 3:
                    self.show_preview(item["image"])
                self.detection_label.config(text=item["summary"])
                self.progress["value"] = item["progress"]
                total = item["total"]
                total_text = str(total) if total > 0 else "?"
                self.status_label.config(
                    text=f"Processing frame {item['frame']}/{total_text}...",
                    fg="blue",
                )
            elif item["kind"] == "finished":
                self.set_running(False)
                self.progress["value"] = 100 if not item["stopped"] else self.progress["value"]
                state = "Stopped after" if item["stopped"] else "Completed"
                cargo_text = (
                    f" | Cargo CSV: {item['cargo_output']}"
                    if item.get("cargo_output") else ""
                )
                self.status_label.config(
                    text=(
                        f"{state} {item['frames']} frames. Video: "
                        f"{item['video_output']} | CSV: {item['csv_output']}"
                        f"{cargo_text}"
                    ),
                    fg="orange" if item["stopped"] else "green",
                )
                if not item["stopped"]:
                    self.load_post_detections(item["csv_output"])
            elif item["kind"] == "error":
                self.set_running(False)
                self.status_label.config(text=f"Processing failed: {item['message']}", fg="red")

        self.root.after(50, self.poll_preview)

    def show_preview(self, image):
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)
        source_width, source_height = image.size
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        displayed_width, displayed_height = image.size
        offset_x = (width - displayed_width) / 2.0
        offset_y = (height - displayed_height) / 2.0
        scale = displayed_width / source_width
        self.display_transform = (
            offset_x,
            offset_y,
            scale,
            source_width,
            source_height,
        )
        self.preview_photo = ImageTk.PhotoImage(image)
        self.canvas.delete("video_frame")
        self.canvas.create_image(
            width // 2,
            height // 2,
            image=self.preview_photo,
            anchor="center",
            tags="video_frame",
        )

    def set_running(self, running):
        self.process_button.config(state="disabled" if running else "normal")
        self.browse_button.config(state="disabled" if running else "normal")
        self.video_entry.config(state="disabled" if running else "normal")
        self.stop_button.config(state="normal" if running else "disabled")
        self.phase_notebook.tab(0, state="disabled" if running else "normal")
        self.phase_notebook.tab(1, state="disabled" if running else "normal")

    def stop_processing(self):
        if self.stop_event is not None:
            self.stop_event.set()
            self.status_label.config(text="Stop requested...", fg="orange")

    def close(self):
        self.crop_export_stop.set()
        self.overlay_export_stop.set()
        self.pause_playback()
        self.pause_post_playback()
        if self.preview_capture is not None:
            self.preview_capture.release()
        if self.stop_event is not None:
            self.stop_event.set()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = OfflineDetectionGUI(root)
    root.mainloop()
