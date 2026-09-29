"""SSH-backed Raspberry Pi camera transport for GUI005."""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import queue
import re
import shutil
import subprocess
import threading

import cv2
import numpy as np


DEFAULT_CAMERA_COMMAND = (
    "if command -v rpicam-vid >/dev/null 2>&1; then "
    "exec rpicam-vid -t 0 --codec mjpeg --width 1280 --height 720 "
    "--framerate 30 -o -; "
    "elif command -v libcamera-vid >/dev/null 2>&1; then "
    "exec libcamera-vid -t 0 --codec mjpeg --width 1280 --height 720 "
    "--framerate 30 -o -; "
    "else echo 'Neither rpicam-vid nor libcamera-vid is installed.' >&2; "
    "exit 127; fi"
)


@dataclass(frozen=True)
class SSHCameraSettings:
    username: str = "sadra"
    host: str = "192.168.50.2"
    port: int = 22

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

    def __init__(self, frame_queue, event_queue, remote_command=DEFAULT_CAMERA_COMMAND):
        self.frame_queue = frame_queue
        self.event_queue = event_queue
        self.remote_command = remote_command
        self.process = None
        self.stop_event = threading.Event()
        self.worker_thread = None
        self.stderr_thread = None

    @staticmethod
    def build_command(
        settings, ssh_executable="ssh", remote_command=DEFAULT_CAMERA_COMMAND
    ):
        settings.validate()
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

    def start(self, settings):
        if self.is_running:
            raise RuntimeError("The SSH camera is already connected.")
        ssh_executable = shutil.which("ssh")
        if not ssh_executable:
            raise RuntimeError(
                "Windows OpenSSH Client was not found. Install it from Optional Features."
            )
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

