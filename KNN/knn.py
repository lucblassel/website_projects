from math import sqrt
import numpy as np
from collections import Counter
from sklearn.datasets import load_iris

class Point:
    def __init__(self, vector, label=""):
        self.vector = vector
        self.label = label

    def __repr__(self):
        return f"Point(vector:{self.vector}, label:{self.label})"

    def distance(self, other):
        return np.linalg.norm(self.vector - other.vector)


class KNN:
    def __init__(self, X, y, k=5):
        self.points = [Point(vector, label) for vector, label in zip(X, y)]
        self.k = k

    def get_distance_to_point(self, point):
        distances = []
        for neighbor in self.points:
            distances.append((neighbor, point.distance(neighbor)))
        return sorted(distances, key=lambda x: x[1])

    def get_nearest_neighbors(self, point):
        return self.get_distance_to_point(point)[:self.k]

    def predict_label(self, point):
        nearest_neighbors = self.get_nearest_neighbors(point)
        counts = Counter([n[0].label for n in nearest_neighbors])
        return max(counts, key=counts.get)

    def get_set_prediction(self, points):
        preds = []
        for point in points:
            preds.append(self.predict_label(Point(point)))
        return np.array(preds)