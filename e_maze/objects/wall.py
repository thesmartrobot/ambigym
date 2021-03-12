import numpy as np
import cv2

import e_maze.objects as o

RED = (0, 0, 255)


class Wall(o.Immovable):
    """Wall object for use in the shadow world environment."""

    def __init__(self, point1, point2):
        self.points = [np.array(point1), np.array(point2)]
        super().__init__(self._position(), False, False)

    def _position(self):
        return sum(self.points) / 2.0

    @property
    def length(self):
        return np.linalg.norm(self.points[1] - self.points[0])

    @property
    def angle(self):
        return np.arctan2(self.points[1][1] - self.points[0][1],
                          self.points[1][0] - self.points[0][0]) % (2.0*np.pi)

    def render(self, image, unit_length):
        pt1 = tuple((self.points[0] * unit_length).astype(int))
        pt2 = tuple((self.points[1] * unit_length).astype(int))
        color = RED
        cv2.line(image, pt1, pt2, color, 2)