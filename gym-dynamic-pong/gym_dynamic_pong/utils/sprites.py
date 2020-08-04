import abc
import math
import random
from typing import Union, Tuple

import numpy as np
from scipy import stats

from . import Rectangle, Line, Point, Shape

__all__ = ['get_critical_angle', 'Paddle', 'Ball', 'Snell', 'Canvas']

EPSILON = 1e-7


def get_critical_angle(s0: float, s1: float) -> Union[float, None]:
    """
    Returns the critical angle if it exists for a ball moving from a medium with velocity `s0` to a medium with
    velocity `s1`. If the critical angle does not exist, returns None.

    :param s0: speed of the initial medium
    :param s1: speed of the final medium
    :return: critical angle or None
    """
    if s0 < s1:
        critical_angle = math.asin(s0 / s1)
    else:
        critical_angle = None
    return critical_angle


class Paddle(Rectangle):
    def __init__(self, height: float, width: float, speed: float, side: str, max_angle: float, visibility: str):
        """

        :param height: The paddle height
        :param width: The paddle width (only matters for rendering)
        :param side: The side the paddle will be on ('left' or 'right')
        :param speed: The units the paddle moves in a single turn
        :param visibility: Whether and how to render the paddle. See `Shape.visibility`
        :param max_angle: The maximum angle at which the paddle can hit the ball
        """
        super().__init__(height=height, width=width, visibility=visibility, render_value=255)
        assert side in ['left', 'right'], f"side must be 'left' or 'right', not {side}"
        assert 0 <= max_angle <= math.pi / 2, f"max angle must be between 0 and pi/2, not {max_angle}"
        self.side = side
        self.speed = speed
        self.max_angle = max_angle

    def up(self):
        self.y += self.speed

    def down(self):
        self.y -= self.speed

    def _get_edges(self) -> Tuple[Line]:
        """
        Only return the field-side edge
        """
        if self.side == 'right':
            return Line((self.left_bound, self.bottom_bound), (self.left_bound, self.top_bound)),
        elif self.side == 'left':
            return Line((self.right_bound, self.bottom_bound), (self.right_bound, self.top_bound)),

    def get_fraction_of_paddle(self, point: Point):
        """
        Computes the fractional distance from the middle of the paddle, normalized by the paddle's height.
        Asserts if the ball was not on the paddle.

        :param point: the point where the ball hit the paddle
        :return: fraction of the paddle
        """
        fraction = (point.y - self.y) / self.height
        fraction = max(min(fraction, 0.5), -0.5)   # clamp to +/- 0.5
        return fraction


class Ball(Rectangle):
    def __init__(self, size: float, max_initial_angle: float, visibility: str, has_volume: bool = False):
        """
        Ball object

        :param has_volume:
        :param size: The size to render the ball
        :param max_initial_angle: The maximum angle the ball can start with
        :param visibility: How to render the ball. See `Shape.visibility`
        :param has_volume: determines whether the ball interacts as a point or as an area
        """
        super().__init__(width=size, height=size, visibility=visibility, render_value=255)
        self.max_initial_angle = max_initial_angle
        self.reset(self.pos, direction='left')
        self.has_volume = has_volume

    def reset(self, position: Union[Tuple[float, float], Point], direction: str = 'right'):
        if direction == 'right':
            self._angle = (2 * random.random() - 1) * self.max_initial_angle
        elif direction == 'left':
            self._angle = math.pi - (2 * random.random() - 1) * self.max_initial_angle
        else:
            raise ValueError(f"direction must be 'left' or 'right', not {direction}")
        self.pos = position

    @property
    def angle(self):
        """
        Angle with respect to the right horizontal
        """
        return self._angle

    @angle.setter
    def angle(self, value):
        self._angle = value % (2 * math.pi)

    @property
    def unit_velocity(self) -> Point:
        x = math.cos(self.angle)
        y = math.sin(self.angle)
        return Point(x, y)

    @unit_velocity.setter
    def unit_velocity(self, value: Union[Tuple[float, float], Point]):
        """
        Sets the angle parameter give a set of (x, y) coordinates.

        :param value: (x, y)
        """
        if isinstance(value, tuple):
            value = Point(*value)
        assert isinstance(value, Point), f"value must be a point, not {type(value)}"
        self.angle = value.angle

    def get_velocity(self, speed: Union[float, int]):
        return self.unit_velocity * speed


class Snell(Rectangle):
    def __init__(self, width, height, speed, change_rate, visibility):
        """
        Rectangular area with a different ball speed.

        :param width: The width of the layer
        :param height: The height of the layer
        :param change_rate: Rate at which the ball speed changes, the standard deviation of the change on each step.
        :param visibility: Whether and how to render the layer. See `Shape.visibility`
        """
        assert change_rate >= 0, "Snell `change_rate` must be non-negative"

        super().__init__(width=width, height=height, visibility=visibility, render_value=(235, 76, 52))
        self.speed = speed
        self._initial_speed = speed
        self.change_rate = change_rate

    def step(self):
        """
        Step the Snell speed using a bounded Gaussian random walk.

        - step with mean 0, standard deviation `self.speed`
        - Clip the speed at `0.5 * self._initial_speed <= self.speed <= 2.0 * self._initial_speed`
        """
        if self.change_rate != 0:
            self.speed += stats.norm(loc=0, scale=self.change_rate).rvs()

            if self.speed < 0.5 * self._initial_speed:
                self.speed = 0.5 * self._initial_speed
            if self.speed > 2.0 * self._initial_speed:
                self.speed = 2.0 * self._initial_speed
        else:
            pass


class TrajectoryBase(abc.ABC):
    def __init__(self, shape: Union[Point, Line, Rectangle], velocity: Point):
        self.shape = shape
        self.velocity = velocity

        self._reference = None
        self.intersection = None
        self.intersected_trajectory = None
        self.intersected_object = None
        self.intersected_edge = None
        self.remaining_speed = None

    def set_intersection(self, point: Point, trajectory_line: Line, obj: Shape, edge: Line):
        assert isinstance(obj, Shape), f"type Shape expected, not {type(obj)}"
        assert isinstance(point, Point), f"type Point expected, not {type(point)}"
        assert isinstance(edge, Line), f"type Line expected, not {type(edge)}"

        self.intersection = point
        self.intersected_trajectory = trajectory_line
        self.remaining_speed = point.l2_distance(trajectory_line.end)
        self.intersected_object = obj
        self.intersected_edge = edge

    def get_center_at_intersection(self) -> Point:
        """
        Get the new center of `self.shape` given that it moved along `intersected_trajectory` to `intersection`

        :return: new center point
        """
        return self._reference + (self.intersection - self.intersected_trajectory.start)

    @property
    def corners(self) -> Tuple[Line, ...]:
        return self.top_left, self.top_right, self.bottom_right, self.bottom_left

    @property
    @abc.abstractmethod
    def center(self) -> Line: ...

    @property
    @abc.abstractmethod
    def top_right(self) -> Line: ...

    @property
    @abc.abstractmethod
    def top_left(self) -> Line: ...

    @property
    @abc.abstractmethod
    def bottom_right(self) -> Line: ...

    @property
    @abc.abstractmethod
    def bottom_left(self) -> Line: ...


class TrajectoryRectangle(TrajectoryBase):
    """
    Compute the trajectory of each corner of the rectangle
    """

    def __init__(self, shape: Rectangle, velocity: Point):
        super(TrajectoryRectangle, self).__init__(shape, velocity)
        assert isinstance(shape, Rectangle)
        self._reference = self.shape.pos

    @property
    def center(self) -> Line:
        """
        Line representing the trajectory of the center of the rectangle
        """
        return Line(self.shape.pos, self.shape.pos + self.velocity)

    @property
    def top_right(self) -> Line:
        """
        Line representing the trajectory of the point on the top right corner of the rectangle
        """
        start = Point(self.shape.right_bound, self.shape.top_bound)
        return Line(start, start + self.velocity)

    @property
    def top_left(self) -> Line:
        """
        Line representing the trajectory of the point on the top left corner of the rectangle
        """
        start = Point(self.shape.left_bound, self.shape.top_bound)
        return Line(start, start + self.velocity)

    @property
    def bottom_right(self) -> Line:
        """
        Line representing the trajectory of the point on the bottom right corner of the rectangle
        """
        start = Point(self.shape.right_bound, self.shape.bottom_bound)
        return Line(start, start + self.velocity)

    @property
    def bottom_left(self) -> Line:
        """
        Line representing the trajectory of the point on the bottom left corner of the rectangle
        """
        start = Point(self.shape.left_bound, self.shape.bottom_bound)
        return Line(start, start + self.velocity)


class TrajectoryLine(TrajectoryRectangle):
    """
    Create a bounding box around the line and compute the trajectory as if it were a rectangle.
    """

    # noinspection PyTypeChecker
    # noinspection PyUnresolvedReferences
    def __init__(self, shape: Line, velocity: Point):
        super(TrajectoryLine, self).__init__(shape, velocity)
        assert isinstance(shape, Line)
        self._reference = self.shape.start

        height = abs(self.shape.start.y - self.shape.end.y)
        width = abs(self.shape.start.x - self.shape.end.x)
        center = Point((self.shape.start.x + self.shape.end.x) / 2,
                       (self.shape.start.y + self.shape.end.y) / 2)
        self.shape = Rectangle(height=height, width=width)
        self.shape.pos = center


class TrajectoryPoint(TrajectoryBase):
    def __init__(self, shape: Point, velocity: Point):
        super(TrajectoryPoint, self).__init__(shape, velocity)
        assert isinstance(shape, Point)
        self._reference = self.shape

    @property
    def corners(self) -> Tuple[Line, ...]:
        return (self._trajectory,)

    @property
    def _trajectory(self) -> Line:
        return Line(self.shape, self.shape + self.velocity)

    @property
    def center(self) -> Line:
        return self._trajectory

    @property
    def top_right(self) -> Line:
        return self._trajectory

    @property
    def top_left(self) -> Line:
        return self._trajectory

    @property
    def bottom_right(self) -> Line:
        return self._trajectory

    @property
    def bottom_left(self) -> Line:
        return self._trajectory


class Trajectory(object):
    def __new__(cls, shape: Shape, velocity: Point):
        if isinstance(shape, Point):
            return TrajectoryPoint(shape, velocity)
        elif isinstance(shape, Line):
            return TrajectoryLine(shape, velocity)
        elif isinstance(shape, Rectangle):
            return TrajectoryRectangle(shape, velocity)
        else:
            raise NotImplementedError(f"No implementation of Trajectory for input shape of type {type(shape)}")


class Canvas(Rectangle):
    action_meanings = {0: 'NOOP',
                       1: 'UP',
                       2: 'DOWN', }
    actions = {k: v for v, k in action_meanings.items()}

    def __init__(self, paddle_l: Paddle, paddle_r: Paddle, ball: Ball, snell: Snell, ball_speed: int, height: int,
                 width: int, their_update_probability: float, refract: bool, uniform_speed: bool):

        super().__init__(height=height, width=width, visibility='none', render_value=0)
        self.pos = self.width / 2, self.height / 2

        assert isinstance(their_update_probability, (float, int)), \
            f"their_update_probability must be numeric, not {type(their_update_probability)}"
        assert 0 <= their_update_probability <= 1, f"{their_update_probability} outside allowed bounds [0, 1]"

        self.their_update_probability = their_update_probability
        self.default_ball_speed = ball_speed

        # Initialize objects
        self.snell = snell
        self.ball = ball
        self.paddle_l = paddle_l
        self.paddle_r = paddle_r
        self.sprites = [self, snell, paddle_l, paddle_r, ball]

        self.uniform_speed = uniform_speed
        self.refract = refract
        self.we_scored = False
        self.they_scored = False

        # score
        self.our_score = 0
        self.their_score = 0

    def register_sprite(self, sprite: Shape):
        assert issubclass(type(sprite), Shape), f"sprite must be subclassed from Shape"
        # noinspection PyTypeChecker
        self.sprites.insert(-1, sprite)  # insert before ball

    @property
    def left_bound(self):
        return 0

    @property
    def right_bound(self):
        return self.width

    @property
    def top_bound(self):
        return self.height

    @property
    def bottom_bound(self):
        return 0

    @property
    def state(self) -> Tuple[float, float, float, float, float, float]:
        """
        Return the basic variable state representation of the environment.
        - ball position (x,y)
        - ball velocity (x,y)
        - opponent paddle y-position
        - our paddle y-position
        """
        velocity = self.ball.get_velocity(self._get_ball_speed())
        return self.ball.pos.x, self.ball.pos.y, velocity.x, velocity.y, self.paddle_l.y, self.paddle_r.y

    # noinspection PyMethodOverriding
    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs masked rendering of objects in `self.sprites`. Priority is determined by the ordering of the list,
        earlier objects will be obscured by later ones.

        :return: (state, rendering)
        """
        state = self._zero_rgb_image(round(self.height), round(self.width))
        rendering = self._zero_rgb_image(round(self.height), round(self.width))

        for sprite in self.sprites[1:]:  # skip self
            sprite_state, sprite_rendering = sprite.to_numpy(self.height, self.width)
            state[sprite_state != 0] = sprite_state[sprite_state != 0]
            rendering[sprite_rendering != 0] = sprite_rendering[sprite_rendering != 0]
        return state, rendering

    def score(self, who):
        """
        Increment the score and reset the ball

        :param who: 'we' or 'they'
        :return: reward
        """
        if who == 'they':
            reward = -1
            self.their_score += 1
        elif who == 'we':
            reward = 1
            self.our_score += 1
        else:
            raise ValueError(f"who must be 'we' or 'they', not {who}")

        self._reset_ball()
        return reward

    def step(self, action):
        self._move_our_paddle(action)
        self._step_their_paddle()
        reward = self._step_ball()
        self._step_snell()
        return reward

    def size(self) -> Tuple[int, int]:
        """
        Return the tuple (height, width) of the canvas dimensions
        """
        return self.height, self.width

    def _step_snell(self) -> None:
        """
        Step the snell layer
        """
        self.snell.step()

    def _reset_ball(self):
        self.ball.reset((self.width / 2, self.height / 2))

    def _move_our_paddle(self, action) -> None:
        """
        Move our paddle according to the provided action

        :param action: the action code
        """
        if not isinstance(action, int):
            action = action.item()  # pops the item if the action is a single tensor
        assert action in [a for a in self.action_meanings.keys()], f"{action} is not a valid action"
        if action == self.actions['UP']:
            if self.paddle_r.top_bound < self.top_bound:
                self.paddle_r.up()
        elif action == self.actions['DOWN']:
            if self.paddle_r.bottom_bound > self.bottom_bound:
                self.paddle_r.down()

    def _step_ball(self, speed: Union[float, int] = None):
        """
        Move the ball to the next position according to the speed of the layer it is in.

        :param speed: used to continue the trajectory of a ball that interacted with an object
        """
        trajectory = self._get_trajectory(speed)

        self._get_first_intersection(trajectory)
        reward = 0
        if trajectory.intersection is None:  # No intersection
            self.ball.pos = trajectory.center.end
        else:
            reward = self._interaction_dispatcher(trajectory)

        return reward

    def _get_trajectory(self, speed) -> TrajectoryBase:
        """
        Get the ball's trajectory

        :param speed: The speed of the starting medium
        :return: trajectory `Line`
        """
        if speed is None:
            speed = self._get_ball_speed()
        if self.ball.has_volume:
            trajectory = Trajectory(self.ball, self.ball.get_velocity(speed))
        else:
            trajectory = Trajectory(self.ball.pos, self.ball.get_velocity(speed))
        return trajectory

    def _interaction_dispatcher(self, trajectory: TrajectoryBase):
        """
        Dispatch data to the appropriate method based on the interaction `obj`.

        :param trajectory: the trajectory of the ball
        """
        reward = 0

        obj = trajectory.intersected_object
        if obj is self:  # border interaction
            reward = self._interact_border(trajectory)
        elif isinstance(obj, Paddle):  # paddle interaction
            self._interact_paddle(trajectory)
        elif isinstance(obj, Snell):
            self._refract(trajectory)

        return reward

    def _interact_paddle(self, trajectory: TrajectoryBase) -> float:
        paddle = trajectory.intersected_object
        paddle_fraction = paddle.get_fraction_of_paddle(trajectory.get_center_at_intersection())
        angle = paddle_fraction * paddle.max_angle
        angle = math.pi - angle if self.ball.unit_velocity.x > 0 else angle

        self.ball.angle = angle
        reward = self._finish_step_ball(trajectory)
        return reward

    def _refract(self, trajectory: TrajectoryBase):
        edge = trajectory.intersected_edge
        if self.refract:
            s0, s1 = self._get_start_and_end_speed(trajectory)

            angle = edge.angle_to_normal(trajectory.center)
            if self._exceeds_critical_angle(angle, s0, s1):
                # TODO: reflect to arbitrary angle (non-vertical interface)
                self._reflect(Point(-1, 1), trajectory)
                return

            new_angle = math.asin(s1 / s0 * math.sin(angle))

            boundary_angle, new_angle = self._adjust_refraction_to_boundary_angle(edge, new_angle)
            new_angle = self._adjust_refraction_to_direction_of_incidence(boundary_angle, new_angle, trajectory)
            self.ball.angle = new_angle

        return self._finish_step_ball(trajectory)

    @staticmethod
    def _exceeds_critical_angle(angle: float, s0: float, s1: float) -> bool:
        """
        Test if the angle exceeds the critical angle

        :param angle: The angle to the normal of the boundary
        :param s0: The speed of the original medium
        :param s1: The speed of the next medium
        :return: True if the angle exceeds the critical angle
        """
        if s1 > s0:  # if the second speed is faster, there is a critical angle
            critical_angle = get_critical_angle(s0, s1)
            if abs(angle) >= critical_angle:
                return True
        return False

    @staticmethod
    def _adjust_refraction_to_direction_of_incidence(boundary_angle: float, new_angle: float,
                                                     trajectory: TrajectoryBase) -> float:
        """
        If the direction of incidence was from the right of the boundary, reflect `new_angle`, otherwise, return
        `new_angle` without modification.

        :param boundary_angle: must be in the first or fourth quadrant
        :param new_angle: The angle to be reflected in the return
        :param trajectory: The angle of the incoming ball in global coordinates
        :return: The (possibly) reflected `new_angle`
        """
        angle = trajectory.center.angle
        assert -math.pi / 2 <= boundary_angle <= math.pi / 2, "boundary_angle should be in first or fourth quadrant"
        # noinspection PyChainedComparisons
        if boundary_angle >= 0 and boundary_angle < angle % (2 * math.pi) < boundary_angle + math.pi:
            new_angle = math.pi - new_angle
        elif (boundary_angle < 0 and
              boundary_angle % (2 * math.pi) + math.pi < angle % (2 * math.pi) < boundary_angle % (
                      2 * math.pi)):
            new_angle = math.pi - new_angle
        return new_angle

    @staticmethod
    def _adjust_refraction_to_boundary_angle(boundary: Line, new_angle: float) -> Tuple[float, float]:
        """
        Compute the rotation of `new_angle` back to global coordinates. Assume incidence from the left side of the
        boundary.

        :param boundary: The boundary `primitives.Line` object
        :param new_angle: The refracted angle normal to the boundary
        :return: The new angle in global coordinates
        """
        # TODO: verify this works with a non-vertical interface

        boundary_angle = boundary.angle % (2 * math.pi)
        if 0 <= boundary_angle < math.pi / 2:  # in the first quadrant
            boundary_angle = boundary_angle
            new_angle = boundary_angle - math.pi / 2 + new_angle
        elif math.pi / 2 <= boundary_angle < math.pi:  # in the second quadrant
            boundary_angle = math.pi - boundary_angle
            new_angle = math.pi / 2 - boundary_angle + new_angle
        elif math.pi <= boundary_angle < 3 * math.pi / 2:  # in the third quadrant
            boundary_angle = math.pi - boundary_angle
            new_angle = boundary_angle - math.pi / 2 + new_angle
        elif 2 * math.pi / 3 <= boundary_angle < 2 * math.pi:  # in the fourth quadrant
            boundary_angle = 2 * math.pi - boundary_angle
            new_angle = math.pi / 2 - boundary_angle - new_angle
        else:
            raise ValueError(f'Unexpected angle {boundary_angle}')
        return boundary_angle, new_angle

    def _get_start_and_end_speed(self, trajectory: TrajectoryBase) -> Tuple[float, float]:
        """
        Get the speed at the start of the trajectory and the speed at the end of the trajectory.

        :param trajectory: The trajectory `primitives.Line` object
        :return: (initial speed, final speed)
        """
        snell = trajectory.intersected_object
        # todo: detect if start is in some other snell layer
        if snell.is_in(trajectory.center.start):
            s0 = snell.speed
            s1 = self.default_ball_speed
        else:
            s0 = self.default_ball_speed
            s1 = snell.speed
        return s0, s1

    def _interact_border(self, trajectory: TrajectoryBase) -> float:
        reward = 0.
        edge = trajectory.intersected_edge

        if edge == self.top_edge or edge == self.bot_edge:
            self._reflect(Point(1, -1), trajectory)
        elif edge == self.left_edge:
            reward = self.score('we')
        elif edge == self.right_edge:
            reward = self.score('they')
        else:
            raise ValueError(f'invalid edge, {edge}')

        return reward

    def _reflect(self, direction: Point, trajectory: TrajectoryBase):
        """
        Multiplies the velocity of the ball by `direction`, continues the path of the ball by calculating the remaining
        speed using trajectory and point.

        :param direction: velocity multiplier
        :param trajectory: The original trajectory of the ball
        """
        self.ball.unit_velocity *= direction
        return self._finish_step_ball(trajectory)

    def _finish_step_ball(self, trajectory: TrajectoryBase):
        """
        Finish the remainder of the trajectory after any interactions.

        :param trajectory: The original trajectory
        :return: reward
        """
        point = trajectory.get_center_at_intersection()
        self.ball.pos = point + self.ball.unit_velocity * EPSILON
        return self._step_ball(trajectory.remaining_speed)

    def _get_first_intersection(self, trajectory: TrajectoryBase):
        """
        Find the first point at which the trajectory interacted with an object.

        :param trajectory: the trajectory of the object
        :return: (shape object interacted with, point of interaction, line object interacted with)
        """
        for trajectory_line in trajectory.corners:
            for o in self.sprites:
                if not isinstance(o, Ball):
                    intersection_result = o.get_intersection(trajectory_line)
                    if intersection_result is not None:
                        edge, point = intersection_result
                        if trajectory.intersection is None:
                            trajectory.set_intersection(point, trajectory_line, o, edge)
                        elif point == trajectory.intersection and trajectory_line == trajectory.intersected_trajectory:
                            raise NotImplementedError("overlapping parallel edges not implemented")
                        elif (point.l2_distance(trajectory_line.start) <
                              trajectory.intersection.l2_distance(trajectory.intersected_trajectory.start)):
                            trajectory.set_intersection(point, trajectory_line, o, edge)

    def _get_ball_speed(self) -> float:
        if self.uniform_speed:
            return self.default_ball_speed
        else:
            if self.ball.is_overlapping(self.snell):
                return self.snell.speed
            else:
                return self.default_ball_speed

    def _step_their_paddle(self):
        """
        Move the opponents paddle. Override this in a subclass to change the behavior.
        """
        if random.random() < self.their_update_probability:
            if self.paddle_l.y < self.ball.y:
                if self.paddle_l.top_bound < self.top_bound:
                    self.paddle_l.up()
            else:
                if self.paddle_l.bottom_bound > self.bottom_bound:
                    self.paddle_l.down()
