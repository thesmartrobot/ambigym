import numpy as np
import cv2

import e_maze.objects as o

LIGHT_BLUE = (230, 216, 173)


class Window(o.Immovable):
    """Window object for use in the shadow world environment."""

    def __init__(self, point1, point2):
        self.points = [np.array(point1), np.array(point2)]
        super().__init__(self._position(), False, True)

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
        color = LIGHT_BLUE
        cv2.line(image, pt1, pt2, color, 2)