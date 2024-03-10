
from pose import Pose
from tracker import Tracker

import threading
import time

import colorsys
import numpy as np

import cv2

class PeopleReporter:
    def __init__(self):
        self.people = dict()

    def update(self, tracker_result):
        objects, disappeared = tracker_result

        to_remove = set()
        for p_id in self.people.keys():
            if p_id not in objects:
                to_remove.add(p_id)

        for p_id in to_remove:
            del self.people[p_id]

        for p_id, landmarks in objects.items():
            if p_id not in self.people:
                # Initial person
                self.people[p_id] = Person(p_id, random_color(), landmarks, 1.0, disappeared)
            else:
                self.people[p_id].landmarks = landmarks
                self.people[p_id].disappeared = disappeared.get(p_id, 50)

        return self.people.values()

class BlobArtist:
    def __init__(self):
        # A blob is a triple of (color, mean, stdev)
        # later we could add intensity
        blobs = []

    def update(self, people):
        noses = [(p.color, p.landmarks[0].x, 0.03) for p in people]
        #lefts = [(p.color, p.landmarks[15].x, 0.03) for p in people if p.landmarks[15].presence > .3 and p.landmarks[15].visibility > .3]
        #rights = [(p.color, p.landmarks[16].x, 0.03) for p in people if p.landmarks[16].presence > .3 and p.landmarks[16].visibility > .3]
        #return noses + lefts + rights

        return noses


class LedVisualizer:
    def __init__(self, people_reporter, num_leds):
        from neopixel import Lights
        self.lights = Lights()

    def update(self, blobs):
        colors = np.array([ b[0] for b in blobs ])
        intensity = np.array([ gauss_curve(blob[1], blob[2], np.linspace(0, 1, self.lights.count)) for blob in blobs ])

        # Create a n x 3 array where n is the number of leds, and 3 are the R,G,B channels
        led_array = (colors[:,:,np.newaxis] * intensity[:,np.newaxis,:]).max(axis=0).T.astype(np.uint8)

        self.lights.set_array(led_array)


class ImshowVisualizer:
    def __init__(self, width = 800, height = 50):
        self.width = width
        self.height = height

    def update(self, blobs):
        image = np.zeros((self.height,self.width,3),np.uint8)

        for blob in blobs:
            color = blob[0]
            intensity = gauss_curve(blob[1], blob[2], np.linspace(0, 1, self.width))

            layer = (color.reshape(3,1) * intensity).astype(np.uint8).T
            layer = np.repeat(layer[:,np.newaxis,:], self.height, axis=1).transpose(1,0,2)
            image = np.maximum(image, layer)
        
        cv2.imshow('foo', image)

class Person:
    def __init__(self, p_id, color, landmarks, intensity, disappeared = 0):
        self.p_id = p_id
        self.color = color
        self.landmarks = landmarks
        self.intensity = intensity
        self.disappeared = disappeared

    def __repr__(self):
        return f"{self.p_id} c {self.color} lm {self.landmarks} i {self.intensity} d {self.disappeared}"

    def __str__(self):
        return f"Person {self.p_id} color {self.color} landmarks {self.landmarks} disappeared {self.disappeared}"


def gauss_curve(mean, stdev, x):
    return np.exp(-0.5 * ((x- mean) / stdev) ** 2)


def random_color():
    h = np.random.rand()
    rgb = colorsys.hsv_to_rgb(h, 1.0, 1.0)

    return np.array([int(v*255) for v in rgb])


class Main:
    def __init__(self, tracker, people_reporter, blob_artist, visualizer):
        self.tracker = tracker
        self.people_reporter = people_reporter
        self.blob_artist = blob_artist
        self.visualizer = visualizer

    def update(self, detection_result, output_image, timestamp_ms):
        result = tracker.update(detection_result)
        #print("tracker:", result)
        result = self.people_reporter.update(result)
        #print("people:", result)
        result = self.blob_artist.update(result)
        #print("blobs:", result)

        self.visualizer.update(result)

if __name__=="__main__":
    tracker = Tracker()
    people_reporter = PeopleReporter()
    blob_artist = BlobArtist()
    #leds = LedVisualizer(50)
    leds = ImshowVisualizer()

    main = Main(tracker, people_reporter, blob_artist, leds)
    Pose(num_poses = 1, callback = main.update).run()

