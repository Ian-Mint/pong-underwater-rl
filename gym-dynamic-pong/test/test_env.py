import unittest
import math

from gym_dynamic_pong.envs import DynamicPongEnv


@unittest.skip
class TestVideoOutputManually(unittest.TestCase):
    def setUp(self) -> None:
        self.env = DynamicPongEnv(default_speed=10, snell_speed=8)

    def test_step_executes_up_manually(self):
        for i in range(50):
            self.env.step(1)
            self.env.render()

    def test_step_executes_down_manually(self):
        for i in range(50):
            self.env.step(1)
            self.env.render()

    def test_big_right_paddle(self):
        self.env.env.ball.angle = 0
        self.env.env.paddle_r.height = 250
        for i in range(200):
            self.env.step(0)
            self.env.render()

    def test_bounce_ball_off_the_top(self):
        self.env.env.ball.angle = math.pi / 4
        self.env.env.paddle_r.height = 250
        for i in range(600):
            self.env.step(0)
            self.env.render()

    def tearDown(self) -> None:
        self.env.close()


class TestEnvironmentBehavior(unittest.TestCase):
    def setUp(self) -> None:
        self.width = 400
        self.height = 300
        self.default_speed = 10
        self.snell_speed = 10
        self.paddle_speed = 3
        self.their_paddle_height = 45
        self.our_paddle_height = 45
        self.their_paddle_probability = 0.2
        self.create_env()
        self.env.step(0)

    def create_env(self):
        pong_env = DynamicPongEnv(max_score=2, width=self.width,
                                  height=self.height,
                                  default_speed=self.default_speed,
                                  snell_speed=self.snell_speed,
                                  our_paddle_speed=self.paddle_speed,
                                  their_paddle_speed=self.paddle_speed,
                                  our_paddle_height=self.our_paddle_height,
                                  their_paddle_height=self.their_paddle_height,
                                  their_update_probability=self.their_paddle_probability)
        self.env = pong_env

    def test_their_score_starts_at_zero(self):
        self.assertEqual(0, self.env.env.their_score)

    def test_our_score_starts_at_zero(self):
        self.assertEqual(0, self.env.env.our_score)

    def test_their_score_increases(self):
        self.env.env.ball.x_pos = self.width - self.default_speed + 1
        self.env.env.ball.y_pos = self.height - 2 * self.default_speed
        self.env.env.ball.angle = 0
        self.env.step(0)
        self.assertEqual(1, self.env.env.their_score)

    def test_our_score_increases(self):
        self.env.env.ball.x_pos = self.default_speed - 1
        self.env.env.ball.y_pos = self.height - 2 * self.default_speed
        self.env.env.ball.angle = math.pi
        self.env.step(0)
        self.assertEqual(1, self.env.env.our_score)

    def test_ball_does_not_get_stuck_in_1000_steps_with_defaults(self):
        self.plausible_ball_motion_tester()

    def test_ball_does_not_get_stuck_in_1000_steps_with_max_paddles(self):
        self.env.env.paddle_r.height = self.height
        self.env.env.paddle_l.height = self.height
        self.plausible_ball_motion_tester()

    def plausible_ball_motion_tester(self):
        self.env.step(0)
        x_prev = self.env.env.ball.x_pos
        y_prev = self.env.env.ball.y_pos
        for i in range(1000):
            self.env.step(0)
            x_pos = self.env.env.ball.x_pos
            y_pos = self.env.env.ball.y_pos

            # ball is too far away from any edge to interact with one and the ball is not at the default start
            if self.default_speed * 4 < x_pos < self.width - self.default_speed * 4 and \
                    self.default_speed * 4 < y_pos < self.height - self.default_speed * 4 and \
                    x_pos != 200:
                # ball is moving right and up
                if 0 < self.env.env.ball.angle < math.pi / 2:
                    self.assertGreater(x_pos, x_prev)
                    self.assertGreater(y_pos, y_prev)
                # ball is moving left and up
                elif math.pi / 2 < self.env.env.ball.angle < math.pi:
                    self.assertLess(x_pos, x_prev)
                    self.assertGreater(y_pos, y_prev)
                # ball is moving left and down
                elif math.pi < self.env.env.ball.angle < math.pi * 3 / 2:
                    self.assertLess(x_pos, x_prev)
                    self.assertLess(y_pos, y_prev)
                # ball is moving right and down
                elif math.pi * 3 / 2 < self.env.env.ball.angle < math.pi * 2:
                    self.assertGreater(x_pos, x_prev)
                    self.assertLess(y_pos, y_prev)

            x_prev = x_pos
            y_prev = y_pos

    def test_ball_hit_exact_upper_edge_and_bounces_correctly(self):
        angle = math.pi / 4
        speed = self.get_ball_speed()
        y_pos = self.height - speed * math.sin(angle)
        x_pos = self.env.env.ball.x_pos

        self.env.env.ball.y_pos = y_pos
        self.env.env.ball.angle = angle

        self.env.step(0)
        self.env.step(0)
        self.assertAlmostEqual(y_pos, self.env.env.ball.y_pos, 2)
        self.assertGreater(self.env.env.ball.x_pos, x_pos)

    def get_ball_speed(self):
        return self.snell_speed if self.env.env.snell.is_in(self.env.env.ball.pos) else self.default_speed

    def test_ball_bounced_off_top_moving_left_over_one_step(self):
        angle = math.pi * 3 / 4
        speed = self.get_ball_speed()
        y_pos = self.height - speed * math.sin(angle) / 2
        x_pos = self.env.env.ball.x_pos

        self.env.env.ball.y_pos = y_pos
        self.env.env.ball.angle = angle

        self.env.step(0)
        self.assertAlmostEqual(y_pos, self.env.env.ball.y_pos, 2)

    def test_ball_bounced_off_bottom_moving_left_over_one_step(self):
        angle = math.pi * 5 / 4
        speed = self.get_ball_speed()
        y_pos = abs(speed * math.sin(angle) / 2)
        x_pos = self.env.env.ball.x_pos

        self.env.env.ball.y_pos = y_pos
        self.env.env.ball.angle = angle

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

        self.env.step(0)
        self.assertAlmostEqual(y_pos, self.env.env.ball.y_pos, 5)
        self.assertGreater(self.env.env.ball.x_pos, x_pos)

    def test_ball_bounced_off_right_paddle(self):
        angle = 0
        y_pos = self.env.env.paddle_r.y_pos
        x_pos = self.width - self.default_speed / 2 - self.env.env.paddle_r.left_bound

        self.env.env.ball.y_pos = y_pos
        self.env.env.ball.x_pos = x_pos
        self.env.env.ball.angle = angle

        self.env.step(0)
        self.assertAlmostEqual(y_pos, self.env.env.ball.y_pos, 0)
        self.assertAlmostEqual(x_pos, self.env.env.ball.x_pos, 1)

    def test_ball_bounced_off_left_paddle(self):
        angle = math.pi
        y_pos = self.height / 2
        x_pos = self.env.env.paddle_l.right_bound + self.env.default_speed / 2

        self.env.env.ball.y_pos = y_pos
        self.env.env.ball.x_pos = x_pos
        self.env.env.ball.angle = angle

        self.env.step(0)
        self.assertAlmostEqual(y_pos, self.env.env.ball.y_pos, 0)
        self.assertAlmostEqual(x_pos, self.env.env.ball.x_pos, 1)

    def test_perfect_opponent_never_is_scored_on(self):
        self.their_update_probability = 1.
        self.default_speed = 2
        self.snell_speed = 2
        self.paddle_speed = 5
        self.our_paddle_height = self.height
        self.create_env()
        for i in range(1000):
            reward = self.env.step(0)[1]
            self.assertEqual(0, reward)

    def test_immobile_opponent_never_moves(self):
        self.their_paddle_probability = 0.
        self.create_env()

        pos = self.env.env.paddle_l.pos
        for i in range(1000):
            self.env.step(0)
            self.assertEqual(pos, self.env.env.paddle_l.pos)

    def tearDown(self) -> None:
        self.env.close()


class TestEnvironmentBehaviorWithRefraction(TestEnvironmentBehavior):
    def setUp(self) -> None:
        super(TestEnvironmentBehaviorWithRefraction, self).setUp()
        self.default_speed = 10
        self.snell_speed = 8
        self.create_env()
        self.env.step(0)


class TestEnvironmentResponse(unittest.TestCase):
    def setUp(self) -> None:
        self.width = 160
        self.height = 160
        self.default_speed = 2
        self.snell_speed = 2
        self.paddle_speed = 3
        self.paddle_height = 30
        pong_env = DynamicPongEnv(max_score=5, width=self.width,
                                  height=self.height,
                                  default_speed=self.default_speed,
                                  snell_speed=self.snell_speed,
                                  our_paddle_speed=self.paddle_speed,
                                  their_paddle_speed=self.paddle_speed,
                                  our_paddle_height=self.paddle_height,
                                  their_paddle_height=self.paddle_height, )
        self.env = pong_env
        self.env.step(0)

    def test_total_reward_after_first_episode_less_than_neg1(self):
        data, reward, episode_over, _ = self.env.step(0)
        total_reward = 0
        while not episode_over:
            data, reward, episode_over, _ = self.env.step(0)
            total_reward += reward

        self.assertLess(total_reward, -1)


if __name__ == '__main__':
    unittest.main()
