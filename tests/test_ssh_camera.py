import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


EXPERIMENTAL_DIR = Path(__file__).resolve().parents[1] / "experimental"
sys.path.insert(0, str(EXPERIMENTAL_DIR))

from ssh_camera import MJPEGFrameParser, SSHCameraClient, SSHCameraSettings


class SSHCameraTests(unittest.TestCase):
    def test_default_ssh_target_and_port(self):
        settings = SSHCameraSettings()
        command = SSHCameraClient.build_command(settings)

        self.assertIn("sadra@192.168.50.2", command)
        self.assertEqual(command[command.index("-p") + 1], "22")
        self.assertIn("BatchMode=yes", command)

    def test_rejects_invalid_connection_settings(self):
        with self.assertRaises(ValueError):
            SSHCameraSettings(username="bad user").validate()
        with self.assertRaises(ValueError):
            SSHCameraSettings(host="bad host!").validate()
        with self.assertRaises(ValueError):
            SSHCameraSettings(port=70000).validate()

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
