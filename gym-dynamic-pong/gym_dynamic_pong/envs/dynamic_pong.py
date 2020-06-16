import os
import time
from typing import Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces

from utils.misc import bool_array_to_rgb
from utils.sprites import Paddle, Ball, Snell, Canvas


class DynamicPongEnv(gym.Env):
    metadata = {'render.modes': ['human', 'png']}

    def __init__(self,
                 max_score=20,
                 width=400,
                 height=300,
                 default_speed=3,
                 snell_speed=3,
                 our_paddle_speed=3,
                 their_paddle_speed=3,
                 our_paddle_height=45,
                 their_paddle_height=45,
                 their_update_probability=0.2,):

        for v in width, height:
            assert isinstance(v, int), "width and height must be integers"

        # configuration
        self.max_score = max_score
        self.width = width
        self.height = height
        self.default_speed = default_speed
        self.snell_speed = snell_speed
        self.our_paddle_speed = our_paddle_speed
        self.their_paddle_speed = their_paddle_speed
        self.our_paddle_height = our_paddle_height
        self.their_paddle_height = their_paddle_height
        self.their_update_probability = their_update_probability

        # initialization
        self._initialize_env()
        self.frame = None
        self.fig = None
        self.ax = None
        self.fig_handle = None
        self.frame_count = 0

        self.observation_space = spaces.Box(low=False, high=True, dtype=np.bool,
                                            shape=(self.env.get_state_size()))
        self.action_space = spaces.Discrete(3)  # initialize discrete action space with 3 actions
        self.ale = ALEInterfaceMock(self.env, self.max_score)

    def step(self, action) -> Tuple[np.ndarray, int, bool, dict]:
        """
        Move the environment to the next state according to the provided action.

        :param action: {0: no-op, 1: up, 2: down}
        :return: (data, reward, episode_over, info)
        """
        reward = self.env.step(action)
        self.frame = self.env.to_numpy()
        return bool_array_to_rgb(self.frame), reward, self.episode_is_over(), {}  # {} is a generic info dictionary

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

    # Display
    def _display_screen(self):
        if self.fig is None:
            self.fig = plt.figure()
            self.ax = self.fig.gca()
            self.fig_handle = self.ax.imshow(self.frame, cmap='gray')
            self.fig.show()
        else:
            self.fig_handle.set_data(self.frame)
        self.ax.set_title(f"{self.env.their_score}                    {self.env.our_score}")
        self.fig.canvas.draw()

    def _save_display_images(self, save_dir):
        """
        Saves the most recent frame as a png image in the directory `save_dir`. If `save_dir` does not exist, it is
        created. Another directory under `save_dir` with the timestamp of the first call of this function is created.
        Numbered frames are stored under this timestamped directory.

        :param save_dir: Directory to save the images in. It will be created if it does not exist.
        """
        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y_%H:%M', t)
        save_dir = os.path.join(save_dir, timestamp)
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        if self.fig is None:
            self.fig = plt.figure()
            self.ax = self.fig.gca()
            self.fig_handle = self.ax.imshow(self.frame, cmap='gray')
        else:
            self.fig_handle.set_data(self.frame)
        self.ax.set_title(f"{self.env.their_score}                    {self.env.our_score}")
        self.fig.canvas.draw()

        path = os.path.join(save_dir, f'{self.frame_count:07d}.png')
        self.frame_count += 1
        self.fig.savefig(path)

    # Sprites
    def _initialize_env(self):
        """
        Initialize the Canvas object containing all the important interactions in the environment.
        """
        self.env = Canvas(
            self._init_paddle('left', self.their_paddle_height, self.their_paddle_speed),
            self._init_paddle('right', self.our_paddle_height, self.our_paddle_speed),
            self._init_ball(),
            Snell(0.25, self.height, self.width, self.snell_speed),
            self.default_speed,
            self.height,
            self.width,
            self.their_update_probability,
        )

    def _init_paddle(self, which_side: str, height, speed) -> Paddle:
        """
        Create a paddle object

        :param which_side: 'left' or 'right'
        :param speed: the number of units the paddle can move in a single frame
        :param height: the height of the paddle
        """
        paddle = Paddle(height, int(0.02 * self.width) + 1, speed, which_side, self.width, self.height)
        paddle.y_pos = self.height / 2
        return paddle

    def _init_ball(self) -> Ball:
        """
        Create a ball object
        """
        ball = Ball(self.height, self.width)
        ball.x_pos = self.width // 2
        ball.y_pos = self.height // 2
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