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

    def test_total_reward_after_first_episode_less_than_neg1(self):
        data, reward, episode_over, _ = self.env.step(0)
        total_reward = 0
        while not episode_over:
            data, reward, episode_over, _ = self.env.step(0)
            total_reward += reward

        self.assertLess(total_reward, -1)