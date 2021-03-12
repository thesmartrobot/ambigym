import numpy as np
import cv2

import e_maze.objects as o
from e_maze.utils import line_segment_line_segment_intersection

GREY = (192, 192, 192)


class Mouse(o.Movable):
    """Mouse object for use in the shadow world environment."""

    def __init__(
        self, position, angle=np.pi / 2.0, speed=0.0, fov=1.1 * np.pi
    ):
        super().__init__(
            position,
            angle=angle,
            speed=speed,
            turn_rate=2 * np.pi,
            max_speed=1,
            min_speed=-0.2,
            mass=0.02,
        )
        self.head_size = 0.1
        self.hip_size = 0.05
        self.fov = fov

    def dimensions(self, angle):
        """
        Radial distance from center to edge at a given angle.
        """
        ori = np.array([0, 0])
        rad = np.array([self.head_size + self.hip_size, 0])

        c = np.cos(self.angle - angle)
        s = np.sin(self.angle - angle)
        R = np.array([[c, -s], [s, c]])
        pt1 = R @ np.array([self.head_size, 0.0])
        pt2 = R @ np.array([-self.head_size, self.hip_size])
        pt3 = R @ np.array([-self.head_size, -self.hip_size])

        i1 = line_segment_line_segment_intersection(ori, rad, pt1, pt2)
        i2 = line_segment_line_segment_intersection(ori, rad, pt2, pt3)
        i3 = line_segment_line_segment_intersection(ori, rad, pt3, pt1)

        dist1 = np.linalg.norm(i1) if i1 is not None else np.inf
        dist2 = np.linalg.norm(i2) if i2 is not None else np.inf
        dist3 = np.linalg.norm(i3) if i3 is not None else np.inf

        return min(dist1, dist2, dist3)

    def render(self, image, unit_length):
        c = np.cos(self.angle)
        s = np.sin(self.angle)
        R = np.array([[c, -s], [s, c]])
        pt1 = (
            self.position + R @ np.array([self.head_size, 0.0])
        ) * unit_length
        pt2 = (
            self.position + R @ np.array([-self.head_size, self.hip_size])
        ) * unit_length
        pt3 = (
            self.position + R @ np.array([-self.head_size, -self.hip_size])
        ) * unit_length
        vertices = np.vstack((pt1, pt2, pt3)).astype(np.int32)
        pts = vertices.reshape((-1, 1, 2))
        color = GREY
        cv2.polylines(image, [pts], True, color, 2)
        cv2.fillPoly(image, [pts], color)

    def time_to_obstacle_collision(self, obstacle):
        # TODO: consider angular speed?
        if self.speed == 0.0:
            return np.inf

        point_time = super().time_to_obstacle_collision(obstacle)

        speed = -self.speed if self.speed < 0.0 else self.speed
        size_time = self.head_size / speed

        return point_time - size_time

    def time_to_entity_collision(self, entity):
        # TODO: consider angular speed?
        return super().time_to_entity_collision(entity)

    def obstacle_collision(self, obstacle):
        """Resolve collision with an obstacle (immovable)."""
        self.bounce_obstacle(obstacle)

    def entity_collision(self, entity):
        """Resolve collision with an entity (movable)."""
        if isinstance(entity, o.Cheese):
            entity.consume()

    def sort_obstacle_endpoint_angles(self, obstacles):
        """
        Sort a list of obstacles (immovable) by clockwise angle
        with angle (facing direction) of the mouse.
        """
        angles = super().sort_obstacle_endpoint_angles(obstacles)
        angles = angles[:-1]
        left = (self.angle - self.fov / 2) % (2.0 * np.pi)
        right = (self.angle + self.fov / 2) % (2.0 * np.pi)
        if left < right:
            filtered_angles = [a for a in angles if left < a < right]
        else:
            left_angles = [a for a in angles if left < a]
            right_angles = [a for a in angles if a < right]
            filtered_angles = left_angles + right_angles
        filtered_angles.insert(0, left)
        filtered_angles.append(right)
        return filtered_angles