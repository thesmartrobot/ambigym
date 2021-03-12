import numpy as np

from e_maze.utils import fuzzy_equal, line_segment_line_segment_intersection, orientation, distance_point_to_line_segment

class Movable():
    def __init__(self, position, angle=0.0, speed=0.0, turn_rate=0.0,
                 max_speed=None, min_speed=None,
                 mass=0, force=None):
        """
        Set force=None to always move at max_speed.
        """
        assert speed is not None
        if max_speed is not None:
            assert max_speed >= 0
            assert -max_speed <= speed <= max_speed
        if min_speed is not None:
            assert min_speed <= 0
            assert min_speed <= speed
        assert turn_rate is not None
        assert max_speed is not None or force is not None

        self.position = position  # current position
        self.angle = angle  # current angle
        self.speed = speed  # current speed
        self.turn_rate = turn_rate  # current turn rate
        self.max_speed = max_speed  # maximum forward speed
        self.min_speed = max_speed if min_speed is None else min_speed  # maximum backwards speed
        self.mass = mass  # mass of movable
        self.force = force  # force exerted when accelerating
        self.consumed = False  # if movable has (been) consumed

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        self._position = position if isinstance(position, np.ndarray) else np.array(position)

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, angle):
        self._angle = angle % (2.0*np.pi)

    @property
    def velocity(self):
        """
        Velocity of the movable.
        """
        return np.array([self.speed * np.cos(self.angle),
                         self.speed * np.sin(self.angle)])

    def dimensions(self, angle):
        """
        To be implemented by subclasses.

        Radial distance from center point to edge of movable at given angle.
        """
        raise NotImplementedError

    def render(self):
        """
        To be implemented by subclasses.

        How to render the movable.
        """
        raise NotImplementedError

    def move(self, dt=0.001, position=None):
        """
        Translational movement for the movable.

        Set dt is None to set position to given position.
        Set dt > 0.0 to make the movable move at its current speed.
        """
        assert dt >= 0 or dt is None, 'dt = ' + str(dt)

        if dt is None:
            if not position:
                raise ValueError("Position must be specified when dt is None.")
            self.position = position
        else:
            self.position += self.velocity * dt

    def turn(self, dt=0.001, angle=0.0):
        """
        Set dt is None to set angle to given angle.
        Set dt > 0.0 to make the movable turn counterclockwise.
        Set dt < 0.0 to make the movable turn clockwise.
        """

        if dt is None:
            self.angle = angle
        else:
            self.angle += (self.turn_rate * dt)

    def accelerate(self, dt=0.001):
        """
        Accelerate for a short time to move forward.
        If max_speed was set, speed will be set to max_speed.
        """
        if self.force is None:
            self.speed = self.max_speed
        else:
            ds = self.force / self.mass * dt
            speed = self.speed + ds
            if self.max_speed is not None \
                    and speed > self.max_speed:
                self.speed = self.max_speed
            else:
                self.speed = speed

    def slow(self, dt=0.001):
        """
        Slow down for a short time to return to a standstill.
        """
        if self.force is None:
            self.speed = 0.0
        else:
            if not self.speed == 0.0:
                ds = self.force / self.mass * dt
                self.speed -= np.sign(self.speed) * ds

    def decelerate(self, dt=0.001):
        """
        Decelerate for a short time to move backwards.
        If min_speed was set, speed will be set to min_speed.
        """
        if self.force is None:
            self.speed = self.min_speed
        else:
            ds = -self.force / self.mass * dt
            speed = self.speed + ds
            if self.min_speed is not None \
                    and speed < self.min_speed:
                self.speed = self.min_speed
            else:
                self.speed = speed

    def angle_to_point(self, point):
        return np.arctan2(point[1] - self.position[1],
                          point[0] - self.position[0]) % (2.0*np.pi)

    def distance_to_point(self, point):
        # unfortunate fix for floating point errors
        return int(np.linalg.norm(point - self.position) * 1e6 + 0.5) / 1e6

    def point_on_obstacle_along_angle(self, obstacle, angle):
        for p in obstacle.points:
            if fuzzy_equal(self.angle_to_point(p), angle):
                return p

        pnt1 = self.position
        c = np.cos(angle)
        s = np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        length = max(self.distance_to_point(obstacle.points[0]),
                     self.distance_to_point(obstacle.points[1])) + 0.5
        pnt2 = R @ np.array([length, 0.0]) + pnt1
        pnt3 = obstacle.points[0]
        pnt4 = obstacle.points[1]

        return line_segment_line_segment_intersection(pnt1, pnt2, pnt3, pnt4)

    def distance_to_obstacle_along_angle(self, obstacle, angle):
        point = self.point_on_obstacle_along_angle(obstacle, angle)

        if point is None:
            return np.inf

        return self.distance_to_point(point)

    def points_of_obstacle_clockwise_of_point(self, obstacle, point):
        clockwise_points = []
        for p in obstacle.points:
            if orientation(self.position, point, p) < 0:
                clockwise_points.append(p)

        return clockwise_points

    def points_of_obstacle_clockwise_of_angle(self, obstacle, angle):
        point = self.point_on_obstacle_along_angle(obstacle, angle)

        if point is None:
            return []

        return self.points_of_obstacle_clockwise_of_point(obstacle, point)

    def closest_distance_to_obstacle_clockwise_of_angle(self, obstacle, angle):
        closest_distance = np.inf

        point = self.point_on_obstacle_along_angle(obstacle, angle)

        if point is None:
            return closest_distance

        points = self.points_of_obstacle_clockwise_of_point(obstacle, point)

        for p in points:
            d = distance_point_to_line_segment(self.position, point, p)
            if d < closest_distance:
                closest_distance = d

        return closest_distance

    def time_to_obstacle_collision(self, obstacle):
        if self.speed == 0.0:
            return np.inf

        if self.speed < 0.0:
            angle = (self.angle + np.pi) % (2.0*np.pi)
            speed = -self.speed
        else:
            angle = self.angle
            speed = self.speed

        distance = self.distance_to_obstacle_along_angle(obstacle, angle)
        return distance / speed

    def bounce_obstacle(self, obstacle):
        if self.force is None:
            self.speed = 0.0
        else:
            theta = obstacle.angle
            theta = theta - np.pi if theta > np.pi else theta
            self.angle = (2*theta - self.angle)

    def distance_to_entity(self, entity):
        return self.distance_to_point(entity.position)

    def time_to_entity_collision(self, entity):
        if self.speed == 0.0:
            return np.inf

        dr = entity.position - self.position
        dv = entity.velocity - self.velocity

        if np.dot(dr, dv) >= 0:
            return np.inf

        angle = self.angle_to_point(entity.position)
        angle2 = (angle + np.pi) % (2.0*np.pi)
        sigma = self.dimensions(angle) + entity.dimensions(angle2)
        d = np.dot(dr, dv)**2 - np.dot(dv, dv) * (np.dot(dr, dr) - sigma**2)

        if d <= 0:
            return np.inf

        return -(np.dot(dr, dv) + np.sqrt(d)) / np.dot(dv, dv)

    def bounce_entity(self, entity):
        """
        Currently unused.
        """
        if self.force is None:
            self.speed = 0.0
            entity.speed = 0.0
        else:
            dr = entity.position - self.position
            dv = entity.velocity - self.velocity
            sigma = self.dimensions(angle) + entity.dimensions(angle2)

            J = 2 * self.mass * entity.mass * np.dot(dr, dv)\
                / sigma / (self.mass + entity.mass)
            vJ = J * dr / sigma
            v1 = self.velocity + vJ / self.mass
            v2 = entity.velocity - vJ / entity.mass

            self.angle = np.arctan2(v1[1], v1[0])
            entity.angle = np.arctan2(v2[1], v2[0])

            speed1 = np.linalg.norm(v1)
            self.speed = speed1
            speed2 = np.linalg.norm(v2)
            entity.speed = speed2

    def obstacle_collision(self, obstacle):
        """
        To be implemented by subclasses.
        """
        raise NotImplementedError

    def entity_collision(self, entity):
        """
        To be implemented by subclasses.
        """
        raise NotImplementedError

    def sort_obstacle_endpoint_angles(self, obstacles):
        points = [p for o in obstacles for p in o.points]
        point_angles = [self.angle_to_point(p) for p in points]
        filtered_angles = list(set(point_angles))
        filtered_angles.sort()
        filtered_angles.append(filtered_angles[0])
        return filtered_angles

    def consume(self):
        self.consumed = True