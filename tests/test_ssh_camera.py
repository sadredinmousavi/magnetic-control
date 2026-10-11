import sys
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np


EXPERIMENTAL_DIR = Path(__file__).resolve().parents[1] / "experimental"
sys.path.insert(0, str(EXPERIMENTAL_DIR))

from ssh_camera import (
    MJPEGFrameParser, SSHCameraClient, SSHCameraSettings,
    build_camera_command, parse_camera_modes, preferred_camera_mode,
)


CAMERA_LIST = """Available cameras
-----------------
0 : imx219 [3280x2464] (/base/soc/camera)
    Modes: 'SRGGB10_CSI2P' : 640x480 [206.65 fps - (1000, 752)/1280x960 crop]
                             1640x1232 [41.85 fps - (0, 0)/3280x2464 crop]
                             1920x1080 [47.57 fps - (680, 692)/1920x1080 crop]
                             3280x2464 [21.19 fps - (0, 0)/3280x2464 crop]
           'SRGGB8' : 1640x1232 [41.85 fps - (0, 0)/3280x2464 crop]
1 : imx477 [4056x3040] (/base/soc/camera2)
    Modes: 'SRGGB12_CSI2P' : 2028x1520 [40.01 fps - (0, 0)/4056x3040 crop]
"""

OV5647_CAMERA_LIST = """Available cameras
-----------------
0 : ov5647 [2592x1944 10-bit GBRG] (/base/soc/i2c0mux/i2c@1/ov5647@36)
    Modes: 'SGBRG10_CSI2P' : 640x480 [58.92 fps - (16, 0)/2560x1920 crop]
                             1296x972 [46.34 fps - (0, 0)/2592x1944 crop]
                             1920x1080 [32.81 fps - (348, 434)/1928x1080 crop]
                             2592x1944 [15.63 fps - (0, 0)/2592x1944 crop]
"""


class SSHCameraTests(unittest.TestCase):
    def test_default_ssh_target_port_and_password(self):
        settings = SSHCameraSettings()
        command = SSHCameraClient.build_command(settings)

        self.assertIn("sadra@192.168.50.2", command)
        self.assertEqual(command[command.index("-P") + 1], "22")
        self.assertEqual(command[command.index("-pw") + 1], "123456789")
        self.assertIn("-batch", command)

    def test_blank_password_uses_openssh_key_authentication(self):
        settings = SSHCameraSettings(password="")
        command = SSHCameraClient.build_command(settings, ssh_executable="ssh")

        self.assertIn("BatchMode=yes", command)
        self.assertNotIn("-pw", command)

    def test_rejects_invalid_connection_settings(self):
        with self.assertRaises(ValueError):
            SSHCameraSettings(username="bad user").validate()
        with self.assertRaises(ValueError):
            SSHCameraSettings(host="bad host!").validate()
        with self.assertRaises(ValueError):
            SSHCameraSettings(port=70000).validate()
        with self.assertRaises(ValueError):
            SSHCameraSettings(password="bad\npassword").validate()

    def test_parse_sensor_modes_including_formats_and_multiple_cameras(self):
        modes = parse_camera_modes(CAMERA_LIST)
        self.assertEqual(len(modes), 6)
        self.assertEqual(modes[0].crop, (1000, 752, 1280, 960))
        self.assertFalse(modes[0].full_field)
        self.assertTrue(modes[1].full_field)
        self.assertEqual(modes[4].mode_spec, "1640:1232:8:U")
        self.assertEqual(modes[5].camera_index, 1)
        self.assertEqual(modes[5].mode_spec, "2028:1520:12:P")

    def test_default_selection_prefers_full_sensor_view(self):
        selected = preferred_camera_mode(parse_camera_modes(CAMERA_LIST))
        self.assertTrue(selected.full_field)
        self.assertEqual(selected.mode_spec, "1640:1232:10:P")

    @patch("ssh_camera.subprocess.run")
    @patch("ssh_camera.SSHCameraClient.find_ssh_executable", return_value="plink")
    def test_ov5647_header_with_bit_depth_and_bayer_order(self, executable, run):
        run.return_value = subprocess.CompletedProcess([], 0, b"", OV5647_CAMERA_LIST.encode())
        modes = SSHCameraClient.list_modes(SSHCameraSettings())
        self.assertEqual(len(modes), 4)
        self.assertEqual(modes[0].camera_name, "ov5647")
        self.assertEqual(modes[0].sensor_size, (2592, 1944))
        self.assertEqual([mode.full_field for mode in modes], [False, True, False, True])
        preferred = preferred_camera_mode(modes)
        self.assertEqual(preferred.mode_spec, "1296:972:10:P")
        command = build_camera_command(SSHCameraSettings(camera_mode=preferred))
        self.assertIn("--mode 1296:972:10:P --width 1296 --height 972", command)

    def test_stream_uses_selected_camera_native_resolution_and_fps(self):
        mode = parse_camera_modes(CAMERA_LIST)[5]
        command = build_camera_command(SSHCameraSettings(camera_mode=mode, framerate=25))
        self.assertIn("--camera 1 --mode 2028:1520:12:P --width 2028 --height 1520", command)
        self.assertIn("--framerate 25", command)
        self.assertNotIn("--width 1280", command)
        self.assertNotIn("--roi", command)

    def test_rejects_fps_above_selected_sensor_mode_limit(self):
        mode = parse_camera_modes(CAMERA_LIST)[3]
        with self.assertRaises(ValueError):
            SSHCameraSettings(camera_mode=mode, framerate=30).validate()

    @patch("ssh_camera.subprocess.run")
    @patch("ssh_camera.SSHCameraClient.find_ssh_executable", return_value="plink")
    def test_query_accepts_camera_list_on_stderr(self, executable, run):
        run.return_value = subprocess.CompletedProcess([], 0, b"", CAMERA_LIST.encode())
        self.assertEqual(len(SSHCameraClient.list_modes(SSHCameraSettings())), 6)
        self.assertIn("--list-cameras", run.call_args.args[0][-1])

    @patch("ssh_camera.subprocess.run")
    @patch("ssh_camera.SSHCameraClient.find_ssh_executable", return_value="plink")
    def test_query_timeout_does_not_expose_password(self, executable, run):
        run.side_effect = subprocess.TimeoutExpired(["plink", "-pw", "123456789"], 20)
        with self.assertRaises(RuntimeError) as error:
            SSHCameraClient.list_modes(SSHCameraSettings())
        self.assertNotIn("123456789", str(error.exception))

    def test_mjpeg_parser_handles_split_and_multiple_frames(self):
        images = []
        for value in (40, 180):
            image = np.full((12, 16, 3), value, dtype=np.uint8)
            success, encoded = cv2.imencode(".jpg", image)
            self.assertTrue(success)
            images.append(encoded.tobytes())

        parser = MJPEGFrameParser()
        stream = b"noise" + images[0] + images[1]
        frames = parser.feed(stream[:17])
        frames.extend(parser.feed(stream[17:]))

        self.assertEqual(len(frames), 2)
        self.assertTrue(all(frame.startswith(b"\xff\xd8") for frame in frames))
        self.assertTrue(all(frame.endswith(b"\xff\xd9") for frame in frames))


if __name__ == "__main__":
    unittest.main()
