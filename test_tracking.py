
from tracking import *
import unittest

# Define a test case class that inherits from unittest.TestCase
class TestPeopleTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = PeopleTracker()

    def test_one_person(self):
        people = self.tracker.update([(10,10)])
        people = self.tracker.update([(11,10)])
        assert len(people) == 1

    def test_two_people_updates_nearest(self):
        people = self.tracker.update([(10,10), (1000, 10)])
        assert len(people) == 2
        people = self.tracker.update([(11,10)])
        assert len(people) == 2
        assert people[0][0] == 11

