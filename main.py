
from pose import Pose
from tracker import Tracker
from neopixel import Lights

import threading
import time

import colorsys
import numpy as np

class Main:
    def __init__(self, num_poses = 2):
        self.tracker = Tracker()
        self.indicators = []
        self.pose = Pose(callback = self.update_tracker, num_poses = num_poses)
        self.lights = Lights()

    def run(self):
        # TODO: start a separate thread that will update the leds
        thread = threading.Thread(target=self.lights_mainloop)
        thread.start()
        self.pose.run()

    def lights_mainloop(self):
        people = dict()
        while True:

            self.update_people(people)

            debug = False
            if debug:
                if len(people) > 0:
                    print(" / ".join([repr(p) for p in people.values()]))

            self.update_lights(people)

    def update_people(self, people):
        # Update states
        objects = self.tracker.objects.copy()
        disappeared = self.tracker.disappeared.copy()

        # NOTE: in edge cases, the keys of disappeared will not contain all keys that are in objects
        # this means the person was removed later, or was only just created.
        # TODO: Think about how to treat this case, or try to avoid it
        #print(objects, disappeared)

        # Remove people that disappeared from tracker
        to_remove = set()
        for p_id in people.keys():
            if p_id not in objects:
                to_remove.add(p_id)

        for p_id in to_remove:
            del people[p_id]

        # Add people we don't know about yet and assign them a color
        for p_id, position in objects.items():
            if p_id not in people:
                # Initial person
                people[p_id] = Person(p_id, random_color(), position, 1.0, disappeared)
            else:
                people[p_id].position = position
                people[p_id].disappeared = disappeared.get(p_id, 50)
                pass
                # TODO: update intensity if they haven't been seen in a while (use tracker.disappeared values)
                # if disappeared = 0 -> increase intensity (max 1)
                # if disappeared > 5 -> decrease intensity (min 0)
                # move light toward position (smoothing, we want to run this with higher frequency)

    def update_lights(self, people):
        # Update lights
        if len(people) == 0:
            colors = np.zeros((self.lights.count, 3))
        else:
            people_colors = np.array([p.color for p in people.values()])
            intensity = np.array([ gauss_curve(person.position, .03, np.linspace(0, 1, self.lights.count)) for person in people.values() ])

            # If we have p people and l leds, then this gives us the color intensity for each led
            colors = (people_colors[:,:,np.newaxis] * intensity[:,np.newaxis,:]).max(axis=0).T

        self.lights.set_array(colors.astype(np.uint8))
        time.sleep(0.03)

    def update_tracker(self, detection_result):
        nose_xs = [ p[0].x for p in detection_result.pose_landmarks ]
        #nose_xs = [ p[0].x for p in detection_result.pose_world_landmarks ]
        self.tracker.update(nose_xs)


class Person:
    def __init__(self, p_id, color, position, intensity, disappeared = 0):
        self.p_id = p_id
        self.color = color
        self.position = position
        self.intensity = intensity
        self.disappeared = disappeared

    def __repr__(self):
        return f"{self.p_id} c {self.color} p {self.position} i {self.intensity} d {self.disappeared}"

    def __str__(self):
        return f"Person {self.p_id} color {self.color} position {self.position} disappeared {self.disappeared}"

def gauss_curve(mean, stdev, x):
    return np.exp(-0.5 * ((x- mean) / stdev) ** 2)

def random_color():
    h = np.random.rand()
    rgb = colorsys.hsv_to_rgb(h, 1.0, 1.0)

    return np.array([int(v*255) for v in rgb])

if __name__=="__main__":
    Main(num_poses = 1).run()
