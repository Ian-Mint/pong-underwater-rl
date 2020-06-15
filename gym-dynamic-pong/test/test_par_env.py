import math
import unittest
import tracemalloc
try:
    from .test_env import TestEnvironmentBehavior, TestEnvironmentResponse
except ImportError:
    from test_env import TestEnvironmentBehavior, TestEnvironmentResponse

from gym_dynamic_pong.envs import ParDynamicPongEnv

tracemalloc.start()


class TestParEnvironmentBehavior(TestEnvironmentBehavior):
    def create_env(self):
        pong_env = ParDynamicPongEnv(max_score=2, width=self.width,
                                     height=self.height,
                                     default_speed=self.default_speed,
                                     snell_speed=self.snell_speed,
                                     our_paddle_speed=self.paddle_speed,
                                     their_paddle_speed=self.paddle_speed,
                                     our_paddle_height=self.our_paddle_height,
                                     their_paddle_height=self.their_paddle_height,
                                     their_update_probability=self.their_paddle_probability)
        self.env = pong_env

    def test_ball_bounced_off_bottom_moving_left_over_one_step(self):
        angle = math.pi * 5 / 4
        speed = self.get_ball_speed()
        y_pos = abs(speed * math.sin(angle) / 2)
        x_pos = self.env.env.ball.x_pos

        self.env.env.ball.y_pos = y_pos
        self.env.env.ball.angle = angle

        self.env._init_step_thread()
        self.env.step(0)
        self.assertAlmostEqual(y_pos, self.env.env.ball.y_pos, 5)
        self.assertLess(self.env.env.ball.x_pos, x_pos)

    def test_ball_bounced_off_bottom_moving_right_over_one_step(self):
        angle = math.pi * 7 / 4
        speed = self.get_ball_speed()
        y_pos = abs(speed * math.sin(angle) / 2)
        x_pos = self.env.env.ball.x_pos

        self.env.env.ball.y_pos = y_pos
        self.env.env.ball.angle = angle

        self.env._init_step_thread()
        self.env.step(0)
        self.assertAlmostEqual(y_pos, self.env.env.ball.y_pos, 5)
        self.assertGreater(self.env.env.ball.x_pos, x_pos)

    def test_ball_bounced_off_left_paddle(self):
        angle = math.pi
        y_pos = self.height / 2
        x_pos = self.env.env.paddle_l.right_bound + self.env.default_speed / 2

        self.env.env.ball.y_pos = y_pos
        self.env.env.ball.x_pos = x_pos
        self.env.env.ball.angle = angle

        self.env._init_step_thread()
        self.env.step(0)
        self.assertAlmostEqual(y_pos, self.env.env.ball.y_pos, 0)
        self.assertAlmostEqual(x_pos, self.env.env.ball.x_pos, 1)

    def test_ball_bounced_off_right_paddle(self):
        angle = 0
        y_pos = self.env.env.paddle_r.y_pos
        x_pos = self.width - self.default_speed / 2 - self.env.env.paddle_r.left_bound

        self.env.env.ball.y_pos = y_pos
        self.env.env.ball.x_pos = x_pos
        self.env.env.ball.angle = angle

        self.env._init_step_thread()
        self.env.step(0)
        self.assertAlmostEqual(y_pos, self.env.env.ball.y_pos, 0)
        self.assertAlmostEqual(x_pos, self.env.env.ball.x_pos, 1)

    def test_ball_bounced_off_top_moving_left_over_one_step(self):
        angle = math.pi * 3 / 4
        speed = self.get_ball_speed()
        y_pos = self.height - speed * math.sin(angle) / 2
        x_pos = self.env.env.ball.x_pos

        self.env.env.ball.y_pos = y_pos
        self.env.env.ball.angle = angle

        self.env._init_step_thread()
        self.env.step(0)
        self.assertAlmostEqual(y_pos, self.env.env.ball.y_pos, 2)

    def test_ball_hit_exact_upper_edge_and_bounces_correctly(self):
        angle = math.pi / 4
        speed = self.get_ball_speed()
        y_pos = self.height - speed * math.sin(angle)
        x_pos = self.env.env.ball.x_pos

        self.env.env.ball.y_pos = y_pos
        self.env.env.ball.angle = angle

        self.env._init_step_thread()
        self.env.step(0)
        self.env.step(0)
        self.assertAlmostEqual(y_pos, self.env.env.ball.y_pos, 2)
        self.assertGreater(self.env.env.ball.x_pos, x_pos)

    def test_our_score_increases(self):
        self.env.env.ball.x_pos = self.default_speed - 1
        self.env.env.ball.y_pos = self.height - 2 * self.default_speed
        self.env.env.ball.angle = math.pi

        self.env._init_step_thread()
        self.env.step(0)
        self.assertEqual(1, self.env.env.our_score)

    def test_their_score_increases(self):
        self.env.env.ball.x_pos = self.width - self.default_speed + 1
        self.env.env.ball.y_pos = self.height - 2 * self.default_speed
        self.env.env.ball.angle = 0

        self.env._init_step_thread()
        self.env.step(0)
        self.assertEqual(1, self.env.env.their_score)


class TestParEnvironmentResponse(TestEnvironmentResponse):
    def setUp(self) -> None:
        self.width = 160
        self.height = 160
        self.default_speed = 2
        self.snell_speed = 2
        self.paddle_speed = 3
        self.paddle_height = 30
        pong_env = ParDynamicPongEnv(max_score=5, width=self.width,
                                     height=self.height,
                                     default_speed=self.default_speed,
                                     snell_speed=self.snell_speed,
                                     our_paddle_speed=self.paddle_speed,
                                     their_paddle_speed=self.paddle_speed,
                                     our_paddle_height=self.paddle_height,
                                     their_paddle_height=self.paddle_height, )
        self.env = pong_env
        self.env.step(0)
