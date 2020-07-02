import math
import os
import random
from typing import Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from gym_dynamic_pong.utils.preprocessing import rgb_array_to_binary
from gym_dynamic_pong.utils.sprites import *


class DynamicPongEnv(gym.Env):
    metadata = {'render.modes': ['human', 'png']}

    def __init__(self,
                 max_score=20,
                 width=400,
                 height=300,
                 default_speed=3,
                 snell_speed=3,
                 snell_width=40,
                 snell_change=0,
                 snell_visible='none',
                 refract=True,
                 uniform_speed=False,
                 our_paddle_speed=3,
                 their_paddle_speed=3,
                 our_paddle_height=45,
                 their_paddle_height=45,
                 their_update_probability=0.2,
                 our_paddle_angle=math.pi / 4,
                 their_paddle_angle=math.pi / 4,
                 ball_size=2,
                 state_type='color', ):

        for v in width, height:
            assert isinstance(v, int), "width and height must be integers"
        assert state_type in ['color', 'binary'], "state type must be 'color' or 'binary'"

        # configuration
        self.max_score = max_score
        self.width = width
        self.height = height
        self.default_speed = default_speed
        self.snell_speed = snell_speed
        self.snell_width = snell_width
        self.snell_change = snell_change
        self.snell_visible = snell_visible
        self.refract = refract
        self.uniform_speed = uniform_speed
        self.our_paddle_speed = our_paddle_speed
        self.their_paddle_speed = their_paddle_speed
        self.our_paddle_height = our_paddle_height
        self.their_paddle_height = their_paddle_height
        self.our_paddle_angle = our_paddle_angle
        self.their_paddle_angle = their_paddle_angle
        self.their_update_probability = their_update_probability
        self.ball_size = ball_size

        # initialization
        self._initialize_env()

        self.state_type = state_type
        self.state = None  # passed to the agent

        # Rendering objects
        self.rendering = None  # passed to the renderer
        self.fig = None
        self.ax = None
        self.fig_handle = None
        self.frame_count = 0

        # todo: this might be wrong, shape might be HxWx3
        self.observation_space = spaces.Box(low=False, high=True, dtype=np.bool,
                                            shape=(self.env.get_state_size()))
        self.action_space = spaces.Discrete(3)  # initialize discrete action space with 3 actions
        self.ale = ALEInterfaceMock(self.env, self.max_score)

    def select_action(self):
        """
        Select action based on the position of the ball
        """
        if random.random() < 0.3:
            if self.env.paddle_r.y < self.env.ball.y:
                if self.env.paddle_r.top_bound < self.env.top_bound:
                    return 1  # up
            else:
                if self.env.paddle_r.bot_bound > self.env.bot_bound:
                    return 2  # down

        return 0  # no-op

    def step(self, action) -> Tuple[np.ndarray, int, bool, dict]:
        """
        Move the environment to the next state according to the provided action.

        :param action: {0: no-op, 1: up, 2: down}
        :return: (data, reward, episode_over, info)
        """
        reward = self.env.step(action)
        self.state, self.rendering = self.env.to_numpy()
        if self.state_type == 'binary':
            self.state = rgb_array_to_binary(self.state)
            self.rendering = rgb_array_to_binary(self.rendering)
        return self.state, reward, self.episode_is_over(), {}  # {} is a generic info dictionary

    def episode_is_over(self):
        """
        :returns: True if the episode is over
        """
        if self.env.their_score == self.max_score or self.env.our_score == self.max_score:
            self.reset()
            return True
        else:
            return False

    def reset(self):
        self._initialize_env()

    def render(self, mode='human', save_dir=None):
        """
        Renders the most recent frame according to the specified mode.
        - human: render to screen using `matplotlib`
        - png: save png images to `save_dir`

        :param mode: 'human' or 'png'
        :param save_dir: directory to save images to in modes other than 'human'
        """
        if mode == 'human':
            self._display_screen()
        elif mode == 'png':
            self._save_display_images(save_dir)

    def close(self):
        self.env = None
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None

    def get_action_meanings(self):
        return self.env.action_meanings

    def _display_screen(self):
        self._update_figure(show=True)

    def _save_display_images(self, save_dir):
        """
        Saves the most recent frame as a png image in the directory `save_dir`. If `save_dir` does not exist, it is
        created. Another directory under `save_dir` with the timestamp of the first call of this function is created.
        Numbered frames are stored under this timestamped directory.

        :param save_dir: Directory to save the images in. It will be created if it does not exist.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self._update_figure(show=False)

        path = os.path.join(save_dir, f'{self.frame_count:d}.png')
        self.frame_count += 1
        self.fig.savefig(path)

    def _update_figure(self, show):
        if self.fig is None:
            self.fig = plt.figure()
            self.ax = self.fig.gca()
            self.fig_handle = self.ax.imshow(self.rendering, cmap='gray')
            if show:
                self.fig.show()
        else:
            self.fig_handle.set_data(self.rendering)
        self.ax.set_title(f"{self.env.their_score}                    {self.env.our_score}")
        self.fig.canvas.draw()

    # Sprites
    def _initialize_env(self):
        """
        Initialize the Canvas object containing all the important interactions in the environment.
        """
        self.env = Canvas(
            self._init_paddle('left', self.their_paddle_height, self.their_paddle_speed, self.their_paddle_angle),
            self._init_paddle('right', self.our_paddle_height, self.our_paddle_speed, self.our_paddle_angle),
            self._init_ball(self.ball_size),
            self._init_snell(self.snell_speed, self.snell_change),
            self.default_speed,
            self.height,
            self.width,
            self.their_update_probability,
            self.refract,
            self.uniform_speed,
        )

    def _init_snell(self, speed: float, change_rate: float):
        # Add one to height so that the boundary does not match the border
        snell = Snell(self.snell_width, self.height + 1, speed, change_rate, visibility=self.snell_visible)
        snell.pos = self.width / 2, self.height / 2
        return snell

    def _init_paddle(self, which_side: str, height: float, speed: float, angle: float) -> Paddle:
        """
        Create a paddle object
        Todo max_angle

        :param which_side: 'left' or 'right'
        :param speed: the number of units the paddle can move in a single frame
        :param height: the height of the paddle
        """
        assert which_side in ['left', 'right'], f"side must be 'left' or 'right', not {which_side}"
        paddle = Paddle(height, int(0.02 * self.width) + 1, speed, which_side, angle, visibility='machine')
        paddle.y = self.height / 2
        if which_side == 'left':
            paddle.x = paddle.width / 2
        if which_side == 'right':
            paddle.x = self.width - paddle.width / 2
        return paddle

    def _init_ball(self, ball_size: float) -> Ball:
        """
        Create a ball object
        """
        max_initial_angle = math.pi / 6
        critical_angle = get_critical_angle(self.snell_speed, self.default_speed)
        if critical_angle is not None:
            if critical_angle < max_initial_angle:
                max_initial_angle = 0.99 * critical_angle
        ball = Ball(size=ball_size, max_initial_angle=max_initial_angle, visibility='machine')
        ball.pos = (self.width / 2, self.height / 2)
        return ball

    def _init_score(self):
        raise NotImplementedError()


class ALEInterfaceMock:
    """
    Object to expose the lives method. There are likely other methods that need to be exposed.
    """

    def __init__(self, env: Canvas, lives: int):
        assert lives > 0, "Number of lives must be > 0"
        self.env = env
        self._lives = lives

    def lives(self) -> int:
        """
        :returns: the minimum of the goals until we and the the goals until the opponent wins
        """
        # TODO: this env object is a copy, does not update when the original does.
        return min(self._lives - self.env.our_score, self._lives - self.env.their_score)
