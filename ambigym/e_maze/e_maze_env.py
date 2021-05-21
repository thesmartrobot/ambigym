import logging
import numpy as np
import cv2

import gym
from gym import spaces
from gym.utils import seeding

from ambigym.e_maze.objects import (
    Movable,
    Immovable,
    Wall,
    Window,
    Cheese,
    Mouse,
)
from ambigym.e_maze.utils import (
    orientation,
    line_segment_circle_intersection,
    line_segment_line_segment_intersection,
    distance_point_to_line_segment,
    fuzzy_equal,
)


LOGGER = logging.getLogger(__name__)

WND_NAME = "maze"
WORLD_SIZE = 3.0  # in meters

WINDOW_SIZE = 960

STATE_W = 96  # less than Atari 160x192
STATE_H = 96
VIDEO_W = WINDOW_SIZE
VIDEO_H = WINDOW_SIZE

FPS = 8

WHITE = (255, 255, 255)


class EMazeEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": FPS,
    }

    def __init__(
        self,
        allow_back=False,
        shadows=True,
        max_length=None,
        player_fov=1.1 * np.pi,
        player_view_distance=None,
        cheese_pos="random",
        close_paths=True,
        reward_type="static",
    ):
        self.seed()
        self.boundaries = []
        self.entities = {}

        self.action_space = spaces.Discrete(n=(4 + allow_back))
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8
        )

        self.viewer = None

        # environment parameters
        # TODO: continuous actions
        self.allow_back = allow_back

        self.shadows = shadows

        self.max_length = max_length

        assert player_fov > 0
        if player_fov <= 2 * np.pi:
            self.player_fov = player_fov
        else:
            self.player_fov = player_fov / 180 * np.pi

        self.player_view_distance = player_view_distance

        assert cheese_pos in ["random", "left", "right"]
        self.cheese_pos = cheese_pos

        self.close_paths = close_paths
        self.paths_closed = False

        assert reward_type in ["static", "decreasing"]
        self.reward_type = reward_type

    def step(self, action):
        dt = 1 / FPS
        curr_time = self.t + dt
        self._action_resolve(action, dt)

        self._step(dt)

        self._action_resolve(0, dt)  # required for discrete actions

        if self.close_paths and not self.paths_closed:
            self._close_paths()

        self.t = curr_time
        self.steps += 1

        # return variables
        observation = self.render(mode="rgb_array")

        found = not bool(
            {
                key: val
                for key, val in self.entities.items()
                if isinstance(val, Cheese)
            }
        )

        if self.reward_type == "static":
            reward = int(found)
        elif self.reward_type == "decreasing":
            reward = -1 if not found else 0

        if self.max_length is not None:
            done = found or self.steps > self.max_length
        else:
            done = found
            
        info = {
            "mouse_pos": self.entities["mouse"].position,
            "mouse_angle": self.entities["mouse"].angle,
            "cheese_pos": self.entities["cheese"].position,
        }

        return observation, reward, done, info

    def _step(self, dt):
        """Advance time by time dt."""
        time_to_collision = np.inf
        collider = None
        collidee = None

        for e in self.entities.values():
            for o in self.boundaries:
                t = e.time_to_obstacle_collision(o)
                if t < time_to_collision:
                    time_to_collision = t
                    collider = e
                    collidee = o

            for e2 in self.entities.values():
                if e2 is e:
                    continue
                t = e.time_to_entity_collision(e2)
                if t < time_to_collision:
                    time_to_collision = t
                    collider = e
                    collidee = e2

        if time_to_collision > dt:
            self._advance(dt)
        else:
            self._advance(time_to_collision)
            self._resolve(collider, collidee)
            self._step(dt - time_to_collision)

    def _action_resolve(self, action, dt):
        """Perform action."""
        if action != 0:
            if action == 1:  # turn left
                self.entities["mouse"].turn(-dt)
            if action == 2:  # turn right
                self.entities["mouse"].turn(dt)
            if action == 3:  # go forward
                self.entities["mouse"].accelerate(dt)
            if self.allow_back and action == 4:  # go backward
                self.entities["mouse"].decelerate(dt)
        else:  # get back to neutral
            self.entities["mouse"].slow(dt)

    def _advance(self, dt):
        """Advance all entities."""
        if dt >= 0:
            for e in self.entities.values():
                e.move(dt=dt)

    def _resolve(self, collider, collidee):
        """Resolve all collisions."""
        if isinstance(collidee, Immovable):
            collider.obstacle_collision(collidee)
        elif isinstance(collidee, Movable):
            collider.entity_collision(collidee)

        self.entities = {
            key: val for key, val in self.entities.items() if not val.consumed
        }

    def _close_paths(self):
        """
        Close paths when the mouse enters a branch.
        Used when close_paths=True.
        """
        mouse = self.entities["mouse"]
        if mouse.position[0] < 1.0:
            self.boundaries.append(Wall((1.0, 2.0), (1.0, 3.0)))
            self.paths_closed = True
        if mouse.position[0] > 2.0:
            self.boundaries.append(Wall((2.0, 2.0), (2.0, 3.0)))
            self.paths_closed = True

    def reset(self):
        self._initialize_terrain()
        self.t = 0
        self.steps = 0
        if self.close_paths:
            self.paths_closed = False
        return self.step(0)[0]

    def _initialize_terrain(self):
        """Initialize the environment."""
        # walls and windows
        self.boundaries = [
            Wall((0.0, 0.0), (0.0, 3.0)),
            Wall((0.0, 3.0), (3.0, 3.0)),
            Wall((3.0, 3.0), (3.0, 0.0)),
            Wall((3.0, 0.0), (0.0, 0.0)),
            Wall((1.0, 0.8), (1.0, 2.0)),
            Wall((2.0, 0.8), (2.0, 2.0)),
            Window((1.0, 0.0), (1.0, 0.8)),
            Window((2.0, 0.0), (2.0, 0.8)),
        ]

        if self.close_paths:
            self.boundaries.append(Wall((0.5, 2.0), (1.0, 2.0)))
            self.boundaries.append(Wall((2.0, 2.0), (2.5, 2.0)))

        # objects
        cheese_x = 0.5
        if self.cheese_pos == "random":
            cheese_x += 2.0 * np.random.randint(2)
        elif self.cheese_pos == "right":
            cheese_x += 2.0

        self.entities = {
            "cheese": Cheese((cheese_x, 0.5)),
            "mouse": Mouse((1.5, 1.5), angle=0, fov=self.player_fov),
        }

    def render(self, mode="human"):
        """
        Different rendering modes:
        'human' will render the environment to the screen.
        'rgb_array' will compute the RGB array only.
        """
        if "t" not in self.__dict__:
            return  # reset() not called yet

        if mode == "human" and self.viewer is None:
            cv2.namedWindow(
                WND_NAME, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_KEEPRATIO
            )
            cv2.resizeWindow(WND_NAME, WINDOW_SIZE, WINDOW_SIZE)
            self.viewer = True

        if mode == "human":
            unit_length = (min(VIDEO_W, VIDEO_H) - 1) / WORLD_SIZE
            image = np.ones((VIDEO_W, VIDEO_H, 3), np.uint8) * 255
        if mode == "rgb_array":
            unit_length = (min(STATE_W, STATE_H) - 1) / WORLD_SIZE
            image = np.ones((STATE_W, STATE_H, 3), np.uint8) * 255

        for boundary in self.boundaries:
            boundary.render(image, unit_length)

        for entity in self.entities.values():
            if not isinstance(entity, Mouse):
                entity.render(image, unit_length)

        if self.shadows:
            visible = self._cast_shadow()
            image = self._render_shadow(visible, image, unit_length)

        if self.player_view_distance is not None:
            image = self._render_view_distance(image, unit_length)

        self.entities["mouse"].render(image, unit_length)

        if mode == "human" and self.viewer:
            cv2.imshow(WND_NAME, image)
            key = cv2.waitKey(1)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        arr = np.asarray(image)
        return arr

    def _cast_shadow(self):
        """Compute the location of shadows cast by boundaries."""
        mouse = self.entities["mouse"]
        nontransparent = [b for b in self.boundaries if not b.transparent]
        # find angles of endpoints of boundaries relative to mouse
        endpoint_angles = mouse.sort_obstacle_endpoint_angles(nontransparent)

        # initialize obstacles and closest_obstacle at first angle
        obstacles = [
            b
            for b in nontransparent
            if mouse.distance_to_obstacle_along_angle(b, endpoint_angles[0])
            < np.inf
        ]
        obstacles.sort(
            key=lambda o: (
                mouse.distance_to_obstacle_along_angle(o, endpoint_angles[0]),
                mouse.closest_distance_to_obstacle_clockwise_of_angle(
                    o, endpoint_angles[0]
                ),  # fix for obstacles starting at same point
            )
        )
        closest_obstacle = obstacles[0]
        # start first wall
        old_point = mouse.point_on_obstacle_along_angle(
            closest_obstacle, endpoint_angles[0]
        )

        # list to hold lit walls
        visible_walls = [mouse.position]

        # loop over endpoint angles
        for angle in endpoint_angles:

            # add any walls that begin at this angle to list
            for boundary in nontransparent:
                these_points = [
                    p
                    for p in boundary.points
                    if fuzzy_equal(mouse.angle_to_point(p), angle)
                ]
                if these_points:
                    other_points = [
                        p
                        for p in boundary.points
                        if not fuzzy_equal(mouse.angle_to_point(p), angle)
                    ]
                    counterclockwise = [
                        orientation(mouse.position, p, o) < 0
                        for p in these_points
                        for o in other_points
                    ]
                    if all(counterclockwise):
                        obstacles.append(boundary)
            obstacles = list(set(obstacles))

            # remove any walls that end at this angle from list
            to_remove = []
            for boundary in obstacles:
                these_points = [
                    p
                    for p in boundary.points
                    if fuzzy_equal(mouse.angle_to_point(p), angle)
                ]
                if these_points:
                    other_points = [
                        p
                        for p in boundary.points
                        if not fuzzy_equal(mouse.angle_to_point(p), angle)
                    ]
                    clockwise = [
                        orientation(mouse.position, p, o) > 0
                        for p in these_points
                        for o in other_points
                    ]
                    if all(clockwise):
                        to_remove.append(boundary)
            obstacles = [o for o in obstacles if o not in to_remove]

            # figure out which wall is now nearest
            obstacles.sort(
                key=lambda o: (
                    mouse.distance_to_obstacle_along_angle(o, angle),
                    mouse.closest_distance_to_obstacle_clockwise_of_angle(
                        o, angle
                    ),  # fix for obstacles starting at same point
                )
            )

            # if nearest wall changed or this is the last iteration
            nearest_wall_changed = obstacles[0] is not closest_obstacle
            is_last_iter = angle == endpoint_angles[-1]
            if nearest_wall_changed or is_last_iter:
                # complete current wall
                new_point = mouse.point_on_obstacle_along_angle(
                    closest_obstacle, angle
                )

                visible_walls.append(old_point)
                visible_walls.append(new_point)

                # and begin a new one
                closest_obstacle = obstacles[0]
                old_point = mouse.point_on_obstacle_along_angle(
                    closest_obstacle, angle
                )

        return visible_walls

    def _render_shadow(self, polygon, image, unit_length):
        """
        Render shadows cast by boundaries.
        Used when shadows=True.
        """
        polygon = np.vstack(polygon)
        mask = np.zeros_like(image, np.uint8) * 255
        color = WHITE
        vertices = (polygon * unit_length).astype(np.int32)
        pts = vertices.reshape((-1, 1, 2))
        cv2.polylines(mask, [pts], True, color, 1)
        cv2.fillPoly(mask, [pts], color)
        return cv2.bitwise_and(image, mask)

    def _render_view_distance(self, image, unit_length):
        """
        Render shadows cast by limited view distance.
        Used when player_view_distance!=None.
        """
        mask = np.zeros_like(image, np.uint8) * 255
        mouse = self.entities["mouse"]
        center = tuple((mouse.position * unit_length).astype(int))
        radius = int(self.player_view_distance * unit_length)
        color = WHITE
        cv2.circle(mask, center, radius, color, -1)
        return cv2.bitwise_and(image, mask)

    def close(self):
        if self.viewer is not None:
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            LOGGER.warning("Program window closed.")
            self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


def test_environment(record_video=False, mode="human", **env_kwargs):
    """
    Test the environment.

    Use arrow keys to move.
    Press s to save an image of the current view.
    Press esc to exit.

    The window closes when the cheese is consumed.

    Keyword arguments:
    record_video -- specify whether to record a video while testing.
    mode -- used for recording, specify which view to record.
    env_kwargs -- environment parameters.

    When record_video=True, a video will be saved in
    the current working folder.
    """
    from pynput import keyboard
    from pynput.keyboard import Key

    def on_press(key):
        if env.steps == 1:
            env.render()
            env.step(0)
            env.render()
            return True

        if key == keyboard.KeyCode.from_char("s"):
            rgb = env.render(mode="rgb_array")
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            cv2.imwrite("env.png", rgb)
            LOGGER.warning("Image saved.")
            return True

        action = 0
        if key == Key.left:
            action = 1
        if key == Key.right:
            action = 2
        if key == Key.up:
            action = 3
        if (
            key == Key.down
            and "allow_back" in env_kwargs
            and env_kwargs["allow_back"]
        ):
            action = 4

        s, _, done, _ = env.step(action)
        r = env.render()
        if record_video and isinstance(r, np.ndarray):
            if mode == "human":
                frame = r
            if mode == "rgb_array":
                frame = s
            img_array.append(frame)
        if done:
            env.close()
            return False
        return True

    def on_release(key):
        if key == Key.esc:
            # Stop listener
            env.close()
            return False
        return True

    if record_video:
        img_array = []

    env = EMazeEnv(**env_kwargs)
    env.reset()
    LOGGER.warning("Press a key to start.")
    with keyboard.Listener(
        on_press=on_press, on_release=on_release
    ) as listener:
        listener.join()
    env.close()

    if record_video:
        from datetime import datetime

        now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        LOGGER.warning("Making video...")
        height, width, layers = img_array[0].shape
        size = (width, height)

        out = cv2.VideoWriter(
            f"cont_mouse_demo_{mode}_{now}.avi",
            cv2.VideoWriter_fourcc(*"FMP4"),
            FPS,
            size,
        )
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
        LOGGER.warning("Done.")


if __name__ == "__main__":
    test_environment()
