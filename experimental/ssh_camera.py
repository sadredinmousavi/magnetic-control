"""SSH-backed Raspberry Pi camera transport for GUI005."""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import math
import os
import queue
import re
import shutil
import subprocess
import threading

import cv2
import numpy as np


def camera_command(options):
    return (
        "if command -v rpicam-vid >/dev/null 2>&1; then "
        f"exec rpicam-vid {options}; "
        "elif command -v libcamera-vid >/dev/null 2>&1; then "
        f"exec libcamera-vid {options}; "
        "else echo 'Neither rpicam-vid nor libcamera-vid is installed.' >&2; "
        "exit 127; fi"
    )


DEFAULT_CAMERA_COMMAND = camera_command("-n -t 0 --codec mjpeg -o -")
LIST_CAMERA_COMMAND = camera_command("--list-cameras")


@dataclass(frozen=True)
class CameraMode:
    camera_index: int
    camera_name: str
    sensor_size: tuple
    width: int
    height: int
    bit_depth: int
    packing: str
    max_fps: float
    crop: tuple

    @property
    def mode_spec(self):
        return f"{self.width}:{self.height}:{self.bit_depth}:{self.packing}"

    @property
    def full_field(self):
        return self.crop == (0, 0, *self.sensor_size)

    @property
    def label(self):
        field = "full sensor view" if self.full_field else "sensor crop"
        return (f"Camera {self.camera_index}: {self.camera_name} | "
                f"{self.width} x {self.height}, {self.bit_depth}-bit {self.packing}, "
                f"up to {self.max_fps:g} fps | {field}")


def parse_camera_modes(output):
    """Read rpicam/libcamera --list-cameras including continuation mode lines."""
    camera = None
    pixel_format = None
    modes = []
    for line in output.splitlines():
        # Newer camera tools also include depth/Bayer metadata in the brackets,
        # e.g. "ov5647 [2592x1944 10-bit GBRG]".
        header = re.match(
            r"\s*(\d+)\s*:\s*(\S+)\s*\[(\d+)x(\d+)(?:\s+[^\]]*)?\]", line
        )
        if header:
            index, name, width, height = header.groups()
            camera = (int(index), name, (int(width), int(height)))
            pixel_format = None
            continue
        format_match = re.search(r"'S[A-Z]+(\d+)(_[A-Z0-9]+)?'", line)
        if format_match:
            pixel_format = (int(format_match[1]), "P" if format_match[2] else "U")
        match = re.search(
            r"(\d+)x(\d+)\s*\[([\d.]+)\s*fps\s*-\s*"
            r"\(\s*(\d+)\s*,\s*(\d+)\s*\)/(\d+)x(\d+)\s*crop\]", line,
        )
        if camera is None or pixel_format is None or match is None:
            continue
        width, height, fps, x, y, crop_w, crop_h = match.groups()
        mode = CameraMode(*camera, int(width), int(height), *pixel_format,
                          float(fps), tuple(map(int, (x, y, crop_w, crop_h))))
        if mode.width > 0 and mode.height > 0 and math.isfinite(mode.max_fps) and mode.max_fps > 0:
            modes.append(mode)
    return modes


def preferred_camera_mode(modes):
    """Prefer the first camera's full view at a modest native streaming size."""
    first_camera = min(mode.camera_index for mode in modes)
    candidates = [mode for mode in modes if mode.camera_index == first_camera]
    full_field = [mode for mode in candidates if mode.full_field]
    if full_field:
        return min(full_field, key=lambda mode: (mode.width * mode.height, -mode.bit_depth))
    return max(candidates, key=lambda mode: (mode.crop[2] * mode.crop[3], mode.bit_depth))


@dataclass(frozen=True)
class SSHCameraSettings:
    username: str = "sadra"
    host: str = "192.168.50.2"
    port: int = 22
    password: str = "123456789"
    camera_mode: CameraMode | None = None
    framerate: float = 30.0

    def validate(self):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", self.username):
            raise ValueError("SSH username contains invalid characters.")
        try:
            ipaddress.ip_address(self.host)
        except ValueError:
            if not re.fullmatch(
                r"(?=.{1,253}\Z)(?!-)[A-Za-z0-9-]+"
                r"(?:\.(?!-)[A-Za-z0-9-]+)*\.?",
                self.host,
            ):
                raise ValueError("Enter a valid Raspberry Pi IP address or hostname.")
        if not 1 <= self.port <= 65535:
            raise ValueError("SSH port must be between 1 and 65535.")
        if "\x00" in self.password or "\r" in self.password or "\n" in self.password:
            raise ValueError("SSH password contains invalid characters.")
        if not math.isfinite(self.framerate) or self.framerate <= 0:
            raise ValueError("Camera frame rate must be positive.")
        if self.camera_mode is not None:
            mode = self.camera_mode
            if (mode.camera_index < 0 or mode.width <= 0 or mode.height <= 0
                    or mode.bit_depth not in (8, 10, 12, 14, 16)
                    or mode.packing not in ("P", "U")):
                raise ValueError("Invalid camera sensor mode.")
            if self.framerate > mode.max_fps:
                raise ValueError(f"Selected camera mode supports up to {mode.max_fps:g} fps.")


def build_camera_command(settings):
    settings.validate()
    options = f"-n -t 0 --codec mjpeg --framerate {settings.framerate:g}"
    mode = settings.camera_mode
    if mode is not None:
        options += (f" --camera {mode.camera_index} --mode {mode.mode_spec}"
                    f" --width {mode.width} --height {mode.height}")
    return camera_command(options + " -o -")


class MJPEGFrameParser:
    """Extract complete JPEG images from an arbitrary byte stream."""

    START = b"\xff\xd8"
    END = b"\xff\xd9"

    def __init__(self):
        self.buffer = bytearray()

    def feed(self, chunk):
        self.buffer.extend(chunk)
        frames = []
        while True:
            start = self.buffer.find(self.START)
            if start < 0:
                if len(self.buffer) > 1:
                    del self.buffer[:-1]
                break
            if start:
                del self.buffer[:start]
            end = self.buffer.find(self.END, 2)
            if end < 0:
                break
            end += len(self.END)
            frames.append(bytes(self.buffer[:end]))
            del self.buffer[:end]
        return frames


class SSHCameraClient:
    """Run a Pi camera command over SSH and publish decoded BGR frames."""

    def __init__(self, frame_queue, event_queue, remote_command=None):
        self.frame_queue = frame_queue
        self.event_queue = event_queue
        self.remote_command = remote_command
        self.process = None
        self.stop_event = threading.Event()
        self.worker_thread = None
        self.stderr_thread = None

    @staticmethod
    def build_command(
        settings, ssh_executable="plink", remote_command=None
    ):
        settings.validate()
        if remote_command is None:
            remote_command = build_camera_command(settings)
        executable_name = os.path.basename(str(ssh_executable)).lower()
        if executable_name in ("plink", "plink.exe"):
            command = [
                ssh_executable,
                "-ssh",
                "-T",
                "-P",
                str(settings.port),
                "-batch",
            ]
            if settings.password:
                command.extend(["-pw", settings.password])
            command.extend(
                [f"{settings.username}@{settings.host}", remote_command]
            )
            return command
        if settings.password:
            raise RuntimeError(
                "Password authentication requires PuTTY plink.exe. Install PuTTY "
                "or clear the password to use OpenSSH key/agent authentication."
            )
        return [
            ssh_executable,
            "-T",
            "-p",
            str(settings.port),
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ServerAliveCountMax=3",
            f"{settings.username}@{settings.host}",
            remote_command,
        ]

    @property
    def is_running(self):
        return self.process is not None and self.process.poll() is None

    @staticmethod
    def find_ssh_executable(settings):
        executable = "plink" if settings.password else "ssh"
        ssh_executable = shutil.which(executable)
        if not ssh_executable:
            if settings.password:
                raise RuntimeError(
                    "PuTTY plink.exe was not found. Install PuTTY or add plink.exe "
                    "to PATH to use password authentication."
                )
            raise RuntimeError(
                "Windows OpenSSH Client was not found. Install it from Optional Features."
            )
        return ssh_executable

    @classmethod
    def list_modes(cls, settings):
        command = cls.build_command(settings, cls.find_ssh_executable(settings), LIST_CAMERA_COMMAND)
        try:
            result = subprocess.run(
                command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=20, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Camera mode query timed out. Check the Pi address and SSH access.") from None
        output = (result.stdout + b"\n" + result.stderr).decode("utf-8", errors="replace")
        # Avoid exposing passwords if a remote error happens to repeat one.
        detail = output.replace(settings.password, "***") if settings.password else output
        if result.returncode:
            raise RuntimeError(f"Could not query camera modes: {detail.strip()[-1200:]}")
        modes = parse_camera_modes(output)
        if not modes:
            raise RuntimeError("The Pi returned no camera sensor modes. " + detail.strip()[-1200:])
        return modes

    def start(self, settings):
        if self.is_running:
            raise RuntimeError("The SSH camera is already connected.")
        ssh_executable = self.find_ssh_executable(settings)
        command = self.build_command(
            settings, ssh_executable, self.remote_command
        )
        self.stop_event.clear()
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            creationflags=creation_flags,
        )
        self.stderr_thread = threading.Thread(
            target=self._read_stderr, daemon=True
        )
        self.worker_thread = threading.Thread(target=self._read_frames, daemon=True)
        self.stderr_thread.start()
        self.worker_thread.start()

    def _read_stderr(self):
        process = self.process
        if process is None or process.stderr is None:
            return
        for raw_line in iter(process.stderr.readline, b""):
            if self.stop_event.is_set():
                break
            message = raw_line.decode("utf-8", errors="replace").strip()
            if message:
                self.event_queue.put(("log", f"Pi camera: {message}"))

    def _read_frames(self):
        process = self.process
        parser = MJPEGFrameParser()
        received_frame = False
        try:
            if process is None or process.stdout is None:
                raise RuntimeError("SSH camera process did not provide a video stream.")
            while not self.stop_event.is_set():
                chunk = process.stdout.read(65536)
                if not chunk:
                    break
                for jpeg in parser.feed(chunk):
                    image = cv2.imdecode(
                        np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    if image is None:
                        continue
                    if not received_frame:
                        received_frame = True
                        self.event_queue.put(("connected",))
                    self._put_latest_frame(image)
            if not self.stop_event.is_set():
                return_code = process.wait()
                detail = (
                    "SSH camera stream ended before the first frame."
                    if not received_frame
                    else "SSH camera stream ended."
                )
                self.event_queue.put(("error", f"{detail} Exit code: {return_code}"))
        except Exception as error:
            if not self.stop_event.is_set():
                self.event_queue.put(("error", str(error)))
        finally:
            if process is self.process:
                self.process = None

    def _put_latest_frame(self, frame):
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass

    def stop(self):
        self.stop_event.set()
        process = self.process
        self.process = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)

