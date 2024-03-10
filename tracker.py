
from collections import OrderedDict
import numpy as np

class Tracker:
    def __init__(self, max_disappeared=50):
        self.next_person_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, landmarks):
        self.objects[self.next_person_id] = landmarks
        self.disappeared[self.next_person_id] = 0
        self.next_person_id += 1

    def deregister(self, person_id):
        del self.objects[person_id]
        del self.disappeared[person_id]

    def handle_present(self, person_id):
        self.disappeared[person_id] = 0

    def handle_disappeared(self, person_id):
        self.disappeared[person_id] += 1

        if self.disappeared[person_id] > self.max_disappeared:
            self.deregister(person_id)

    def update(self, detection_result):
        landmarks = detection_result.pose_landmarks
        if len(landmarks) == 0:
            for person_id in list(self.disappeared.keys()):
                self.handle_disappeared(person_id)

            return self.objects, self.disappeared

        if len(self.objects) == 0:
            for i in range(0, len(landmarks)):
                self.register(landmarks[i])

        else:
            person_ids = list(self.objects.keys())
            person_landmarks = list(self.objects.values())

            D = np.zeros((len(person_landmarks), len(landmarks)), dtype=int)
            # Calculates the distance from each person to each detection result
            for i in range(len(person_landmarks)):
                for j in range(len(landmarks)):
                    D[i, j] = np.linalg.norm(person_landmarks[i][0].x - landmarks[j][0].x)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                person_id = person_ids[row]
                self.objects[person_id] = landmarks[col]
                self.handle_present(person_id)

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    person_id = person_ids[row]
                    self.handle_disappeared(person_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])

        return self.objects, self.disappeared
