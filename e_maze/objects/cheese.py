import numpy as np
import cv2

import e_maze.objects as o

YELLOW = (0, 255, 255)


class Cheese(o.Movable):
    """Cheese object for use in the shadow world environment."""

    def __init__(self, position):
        super().__init__(position, max_speed=0.0, mass=5)
        self.radius = 0.2

    def dimensions(self, angle):
        return self.radius

    def render(self, image, unit_length):
        center = tuple((self.position * unit_length).astype(int))
        radius = int(self.radius * unit_length)
        color = YELLOW
        cv2.circle(image, center, radius, color, -1)

    def obstacle_collision(self, obstacle):
        self.bounce_obstacle(obstacle)

    def entity_collision(self, entity):
        if isinstance(entity, o.Mouse):
            self.consume()