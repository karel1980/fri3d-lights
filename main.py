
from pose import Pose
from tracker import Tracker
from neopixel import Lights

import threading
import time

import colorsys
import numpy as np

class Main:
    def __init__(self):
        self.tracker = Tracker()
        self.indicators = []
        self.pose = Pose(callback = self.update_tracker)
        self.lights = Lights()

    def run(self):
        # TODO: start a separate thread that will update the leds
        thread = threading.Thread(target=self.update_lights)
        thread.start()
        self.pose.run()

    def update_lights(self):
        i = 0
        people = dict()
        while True:
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
                    people[p_id] = Person(p_id, random_color(), position, 1.0)
                else:
                    pass
                    # TODO
                    # if disappeared = 0 -> increase intensity (max 1)
                    # if disappeared > 5 -> decrease intensity (min 0)
                    # move light toward position (smoothing, we want to run this with higher frequency)

            # Update lights
            if len(people) == 0:
                colors = np.zeros((self.lights.count, 3))
            else:
                # TODO: fade people colors i they haven't been seen in a while (use tracker.disappeared values)
                people_colors = np.array([p.color for p in people.values()])
                intensity = np.array([ gauss_curve(objects[p_id], .03, np.linspace(0, 1, self.lights.count)) for p_id in people.keys() ])

                # If we have p people and l leds, then this gives us the color intensity for each led
                colors = (people_colors[:,:,np.newaxis] * intensity[:,np.newaxis,:]).max(axis=0).T

                #print((intensity*100).astype(np.int32))

            #print(colors[0])
            self.lights.set_array(colors.astype(np.uint8))

            i+=1
            time.sleep(0.01)

    def update_tracker(self, detection_result):
        nose_xs = [ p[0].x for p in detection_result.pose_landmarks ]
        #nose_xs = [ p[0].x for p in detection_result.pose_world_landmarks ]
        self.tracker.update(nose_xs)


    def update_leds(self):
        #print(self.tracker.objects.copy())
        pass
        # iterate over objects from tracker,
        # etc


class Person:
    def __init__(self, p_id, color, position, intensity):
        self.p_id = p_id
        self.color = color
        self.position = position
        self.intensity = intensity

def gauss_curve(mean, stdev, x):
    return np.exp(-0.5 * ((x- mean) / stdev) ** 2)

def random_color():
    h = np.random.rand()
    rgb = colorsys.hsv_to_rgb(h, 1.0, 1.0)

    return np.array([int(v*255) for v in rgb])

if __name__=="__main__":
    Main().run()
