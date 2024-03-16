
from collections import OrderedDict
import numpy as np

class Tracker:
    def __init__(self, distance_metric, max_disappeared=50):
        self.distance_metric = distance_metric
        self.max_disappeared = max_disappeared
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

    def register(self, features):
        self.objects[self.next_object_id] = features
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def handle_present(self, object_id):
        self.disappeared[object_id] = 0

    def handle_disappeared(self, object_id):
        self.disappeared[object_id] += 1

        if self.disappeared[object_id] > self.max_disappeared:
            self.deregister(object_id)

    def update(self, features):
        if len(features) == 0:
            for object_id in list(self.disappeared.keys()):
                self.handle_disappeared(object_id)

            return self.objects, self.disappeared

        if len(self.objects) == 0:
            for i in range(0, len(features)):
                self.register(features[i])

        else:
            object_ids = list(self.objects.keys())
            object_features = list(self.objects.values())

            #TODO: do distance calculation based on more than just nose x coordinates

            D = np.zeros((len(object_features), len(features)), dtype=int)
            # Calculates the distance from each tracked object to each detection result
            for i in range(len(object_features)):
                for j in range(len(features)):
                    D[i, j] = self.distance_metric(object_features[i], features[j])

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = features[col]
                self.handle_present(object_id)

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.handle_disappeared(object_id)
            else:
                for col in unused_cols:
                    self.register(features[col])

        return self.objects, self.disappeared
