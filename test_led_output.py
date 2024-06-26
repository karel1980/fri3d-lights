from main import LedOutputPlugin, Person, bell_curve, scaled_bell_curve
from dataclasses import dataclass

import numpy as np

import unittest

class FakeLedStrip():
    def __init__(self, num_leds = 2):
        self.num_leds = num_leds
    def set_all_pixels(self, color):
        pass
    def set_array(self, colors):
        pass

class FakeDetectionResult:
    def __init__(self, landmarks):
        self.pose_landmarks = landmarks
        self.pose_world_landmarks = landmarks

@dataclass
class LandMark:
    x: int
    y: int
    z: int

class LedOutputPluginTest(unittest.TestCase):

    def test_bell_curve(self):
        self.assertEqual(bell_curve(0,0,1), 1)
        self.assertEqual(bell_curve(0,0,.1), 1)
        self.assertEqual(bell_curve(0.5,0.5,99), 1)

    def test_calculate_led_colors(self):
        plugin = LedOutputPlugin(FakeLedStrip())
        blobs = [Person(0, None, (0, 0.1), (0, 255, 0))]
        self.assertEqual(plugin.calculate_led_colors(blobs).tolist(), np.array([[0,0,0],[0,255,0]]).tolist())


# Running the tests
if __name__ == '__main__':
    unittest.main()

