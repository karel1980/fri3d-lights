
from tracker import *
import unittest

from dataclasses import dataclass

@dataclass
class LandMark:
    x: int
    y: int
    z: int

class FakeDetectionResult:
    def __init__(self, landmarks=[]):
        self.pose_landmarks = landmarks
        self.pose_world_landmarks = landmarks

def fake_detection_result(*xs):
    return FakeDetectionResult([[LandMark(x, 0, 0)] for x in xs])

# Define a test case class that inherits from unittest.TestCase
class TestTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = Tracker(self.distance_metric)

    def distance_metric(self, a, b):
        return np.linalg.norm(a[0].x - b[0].x)

    def test_one_person(self):
        obj, det = self.tracker.update(fake_detection_result(10,10).pose_landmarks)
        obj, det = self.tracker.update(fake_detection_result(11,10).pose_landmarks)
        self.assertEqual(len(obj), 2)

    def test_two_people_updates_nearest(self):
        obj, det = self.tracker.update(fake_detection_result(10, 1000).pose_landmarks)
        self.assertEqual(len(obj), 2)
        obj, det = self.tracker.update(fake_detection_result(11).pose_landmarks)
        self.assertEqual(len(obj), 2)

