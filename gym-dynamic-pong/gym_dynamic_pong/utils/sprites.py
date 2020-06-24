import math
import random
from typing import Union, Tuple, List

import numpy as np

from . import Rectangle, Line, Point, Shape

EPSILON = 1e-7


class Paddle(Rectangle):
    def __init__(self, height: float, width: float, speed: float, side: str, max_angle: float):
        """

        :param height: The paddle height
        :param width: The paddle width (only matters for rendering)
        :param side: The side the paddle will be on ('left' or 'right')
        :param speed: The units the paddle moves in a single turn
        :param max_angle: The maximum angle at which the paddle can hit the ball
        """
        super().__init__(height=height, width=width)
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
            return Line((self.left_bound, self.bot_bound), (self.left_bound, self.top_bound)),
        elif self.side == 'left':
            return Line((self.right_bound, self.bot_bound), (self.right_bound, self.top_bound)),

    def get_fraction_of_paddle(self, point: Point):
        """
        Computes the fractional distance from the middle of the paddle, normalized by the paddle's height.
        Asserts if the ball was not on the paddle.

        :param point: the point where the ball hit the paddle
        :return: fraction of the paddle
        """
        fraction = (point.y - self.y) / self.height
        assert -0.5 <= fraction <= 0.5, "The ball was not on the paddle"
        return fraction


class Ball(Rectangle):
    def __init__(self):
        super().__init__(width=2, height=2)
        self._angle = math.pi - (random.random() - 0.5) * (math.pi / 3)

    def reset(self, position: Union[Tuple[float, float], Point]):
        self._angle = (random.random() - 0.5) * (math.pi / 3)
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


class Snell(Rectangle):
    def __init__(self, width, height, speed):
        """
        area indicating ball speed.
        :return:
        """
        super().__init__(width=width, height=height)
        self.speed = speed


class Canvas(Rectangle):
    action_meanings = {0: 'NOOP',
                       1: 'UP',
                       2: 'DOWN', }
    actions = {k: v for v, k in action_meanings.items()}

    def __init__(self, paddle_l: Paddle, paddle_r: Paddle, ball: Ball, snell: Snell, ball_speed: int, height: int,
                 width: int, their_update_probability: float, paddle_angle: float, **kwargs):

        super().__init__(height=height, width=width, **kwargs)
        self.pos = self.width / 2, self.height / 2

        assert isinstance(their_update_probability, (float, int)),\
            f"their_update_probability must be numeric, not {type(their_update_probability)}"
        assert 0 <= their_update_probability <= 1, f"{their_update_probability} outside allowed bounds [0, 1]"

        self.their_update_probability = their_update_probability
        self.default_ball_speed = ball_speed

        # Initialize objects
        self.snell = snell
        self.ball = ball
        self.paddle_l = paddle_l
        self.paddle_r = paddle_r

        self.we_scored = False
        self.they_scored = False

        # score
        self.our_score = 0
        self.their_score = 0

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
    def bot_bound(self):
        return 0

    def get_objects(self):
        return self, self.snell, self.paddle_r, self.paddle_l

    # noinspection PyMethodOverriding
    def to_numpy(self) -> np.ndarray:
        out = np.zeros((round(self.height), round(self.width)), dtype=np.bool)

        for sprite in (self.ball, self.paddle_l, self.paddle_r):
            out |= sprite.to_numpy(self.height, self.width)
        return out

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
        return self._step_ball()

    def get_state_size(self) -> Tuple[int, int]:
        """
        Return the tuple (height, width) of the canvas dimensions
        """
        return self.height, self.width

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
            if self.paddle_r.bot_bound > self.bot_bound:
                self.paddle_r.down()

    def _step_ball(self, speed: Union[float, int] = None):
        """
        Move the ball to the next position according to the speed of the layer it is in.

        :param speed: used to continue the trajectory of a ball that interacted with an object
        """
        if speed is None:
            speed = self._get_ball_speed()
        new_pos = tuple(speed * v + x for x, v in zip(self.ball.pos, self.ball.unit_velocity))
        # noinspection PyTypeChecker
        trajectory = Line(self.ball.pos, new_pos)

        result = self._get_first_intersection(trajectory)
        reward = 0
        if result is None:  # No intersection
            self.ball.pos = new_pos
        else:
            reward = self._interaction_dispatcher(result, trajectory)

        return reward

    def _interaction_dispatcher(self, interaction_result: List, trajectory: Line):
        """
        Dispatch data to the appropriate method based on the interaction `obj`.

        :param trajectory: the trajectory of the ball
        :param obj: An object in the canvas
        :param point: the point of interaction
        """
        reward = 0
        if len(interaction_result) == 1:
            obj, point, edge = interaction_result.pop()
            assert isinstance(obj, Shape), f"type Shape expected, not {type(obj)}"
            assert isinstance(point, Point), f"type Point expected, not {type(point)}"
            assert isinstance(edge, Line), f"type Line expected, not {type(edge)}"

            if obj is self:  # border interaction
                reward = self._interact_border(point, edge, trajectory)
            elif isinstance(obj, Paddle):  # paddle interaction
                self._interact_paddle(obj, point, trajectory)
            elif isinstance(obj, Snell):
                self._refract(obj, point, edge, trajectory)
        else:
            raise NotImplementedError("Shared boundary not yet implemented")

        return reward

    def _interact_paddle(self, paddle: Paddle, point: Point, trajectory: Line) -> float:
        paddle_fraction = paddle.get_fraction_of_paddle(point)
        angle = paddle_fraction * paddle.max_angle
        angle = math.pi - angle if self.ball.unit_velocity.x > 0 else angle

        self.ball.angle = angle
        reward = self._finish_step_ball(point, trajectory)
        return reward

    def _refract(self, obj: Snell, point: Point, edge: Line, trajectory: Line):
        s0, s1 = self._get_start_and_end_speed(obj, trajectory)

        angle = edge.angle_to_normal(trajectory)
        if self._exceeds_critical_angle(angle, s0, s1):
            # TODO: reflect to arbitrary angle (non-vertical interface)
            self._reflect(Point(-1, 1), point, trajectory)
            return

        new_angle = math.asin(s1 / s0 * math.sin(angle))

        boundary_angle, new_angle = self._adjust_refraction_to_boundary_angle(edge, new_angle)
        new_angle = self._adjust_refraction_to_direction_of_incidence(boundary_angle, new_angle, trajectory)
        self.ball.angle = new_angle

        return self._finish_step_ball(point, trajectory)

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
            critical_angle = math.asin(s0 / s1)
            if abs(angle) >= critical_angle:
                return True
        return False

    @staticmethod
    def _adjust_refraction_to_direction_of_incidence(boundary_angle, new_angle, trajectory):
        """
        If the direction of incidence was from the right of the boundary, reflect `new_angle`, otherwise, return
        `new_angle` without modification.

        :param boundary_angle: must be in the first or fourth quadrant
        :param new_angle: The angle to be reflected in the return
        :param trajectory: The angle of the incoming ball in global coordinates
        :return: The (possibly) reflected `new_angle`
        """
        assert -math.pi / 2 <= boundary_angle <= math.pi / 2, "boundary_angle should be in first or fourth quadrant"
        if boundary_angle >= 0 and boundary_angle < trajectory.angle % (2 * math.pi) < boundary_angle + math.pi:
            new_angle = math.pi - new_angle
        elif (boundary_angle < 0 and
              boundary_angle % (2 * math.pi) + math.pi < trajectory.angle % (2 * math.pi) < boundary_angle % (
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

    def _get_start_and_end_speed(self, snell: Snell, trajectory: Line) -> Tuple[float, float]:
        """
        Get the speed at the start of the trajectory and the speed at the end of the trajectory.

        :param trajectory: The trajectory `primitives.Line` object
        :return: (initial speed, final speed)
        """
        # todo: detect if start is in some other snell layer
        if snell.is_in(trajectory.start):
            s0 = snell.speed
            s1 = self.default_ball_speed
        else:
            s0 = self.default_ball_speed
            s1 = snell.speed
        return s0, s1

    def _interact_border(self, point: Point, edge: Line, trajectory: Line) -> float:
        reward = 0.
        if edge == self.top_edge or edge == self.bot_edge:
            self._reflect(Point(1, -1), point, trajectory)
        elif edge == self.left_edge:
            reward = self.score('we')
        elif edge == self.right_edge:
            reward = self.score('they')
        else:
            raise ValueError(f'invalid edge, {edge}')

        return reward

    def _reflect(self, direction: Point, point: Point, trajectory: Line):
        """
        Multiplies the velocity of the ball by `direction`, continues the path of the ball by calculating the remaining
        speed using trajectory and point.

        :param direction: velocity multiplier
        :param point: The point of interaction
        :param trajectory: The original trajectory of the ball
        """
        self.ball.unit_velocity *= direction
        return self._finish_step_ball(point, trajectory)

    def _finish_step_ball(self, point, trajectory):
        self.ball.pos = point + self.ball.unit_velocity * EPSILON
        remaining_speed = point.l2_distance(trajectory.end)
        return self._step_ball(remaining_speed)

    def _get_first_intersection(self, trajectory: Line) -> Union[List[Tuple[Shape, Point, Line]], None]:
        """
        Find the first point at which the trajectory interacted with an object.

        :param trajectory: the trajectory of the object
        :return: (shape object interacted with, point of interaction, line object interacted with)
        """
        result = None

        for o in self.get_objects():
            intersection_result = o.get_intersection(trajectory)
            if intersection_result is not None:
                edge, intersection = intersection_result
                if result is None:
                    result = [(o, intersection, edge)]
                elif intersection == result[0][1]:  # we have a shared boundary
                    result = result.append((o, intersection, edge))
                elif trajectory.point1_before_point2(intersection, result[0][1]):
                    result = [(o, intersection, edge)]
        return result

    def _get_ball_speed(self) -> float:
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
                if self.paddle_l.bot_bound > self.bot_bound:
                    self.paddle_l.down()