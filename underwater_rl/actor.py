import argparse
import io
import logging
import math
import os
import pickle
import random
import shutil
import time
from copy import deepcopy
from itertools import count
from typing import Union, Dict, Tuple, List

import gym
import numpy as np
import torch
from PIL import Image
from torch import nn as nn, multiprocessing as mp

try:
    from underwater_rl.base import BaseWorker, ParamPipe, Transition, HistoryElement, DEVICE
    from underwater_rl.utils import get_logger_from_process, convert_images_to_video
    from underwater_rl.wrappers import LazyFrames, make_env
except ImportError:
    from base import BaseWorker, ParamPipe, Transition, HistoryElement
    from main import dispatch_make_env, ACTOR_UPDATE_INTERVAL, MAX_STEPS_PER_EPISODE, N_ACTIONS, DEVICE
    from utils import get_logger_from_process, HistoryElement, Transition, convert_images_to_video
    from wrappers import LazyFrames, make_env


MAX_STEPS_PER_EPISODE = 50_000
N_ACTIONS = 3
ACTOR_UPDATE_INTERVAL = 1000


class State:
    def __init__(self, state: Union[torch.Tensor, None], device: str = DEVICE):
        """
        If device is different from `state.device`, create a copy of `state` on specified device at `State.cuda`.

        The assumption is that the passed `state` is on the CPU and `device` will either be a cuda device, or just cpu
        if not cuda device is available. In the second case, no copy is made.
        :param state: State on cpu (no check is made to enforce this)
        :param device: Device to copy `state` to; e.g. cuda, cpu.
        """
        if state is None:
            self.cpu, self.cuda = None, None
        else:
            self.cpu = state
            self.cuda = state.to(device, non_blocking=True)


def get_state(obs: LazyFrames, device: str) -> State:
    """
    Return a `State` object given an environment observation and a device to push the state to

    :param obs: A state object wrapped with LazyFrames
    :param device: e.g. cpu, cuda
    :return: `State` object
    """
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    state = state.unsqueeze(0)
    return State(state, device)


def dispatch_make_env(args):
    r"""
    Make a new pong environment

    :param args: The namespace returned by argparse
    """
    env = gym.make(
        "gym_dynamic_pong:dynamic-pong-v1",
        max_score=20,
        width=args.width,
        height=args.height,
        default_speed=args.ball,
        snell_speed=args.snell,
        snell_width=args.snell_width,
        snell_change=args.snell_change,
        snell_visible=args.snell_visible,
        refract=not args.no_refraction,
        uniform_speed=args.uniform_speed,
        our_paddle_speed=args.paddle_speed,
        their_paddle_speed=args.paddle_speed,
        our_paddle_height=args.paddle_length,
        their_paddle_height=args.paddle_length,
        our_paddle_angle=math.radians(args.paddle_angle),
        their_paddle_angle=math.radians(args.paddle_angle),
        their_update_probability=args.update_prob,
        ball_size=args.ball_size,
        ball_has_volume=args.ball_volume,
        state_type=args.state,
    )

    if args.network == 'lstm':
        env = make_env(env, stack_frames=False, episodic_life=True, clip_rewards=True, max_and_skip=False)
    else:
        env = make_env(env, stack_frames=True, episodic_life=True, clip_rewards=True, max_and_skip=True)
    return env


class Actor(BaseWorker):
    r"""
    +-----+        +-------+
    |Actor|------->|Encoder|
    +-----+        +-------+
    """
    counter = 0  # count the number of actor instances

    def __init__(self,
                 model: nn.Module,
                 n_episodes: int,
                 render_mode: Union[str, bool],
                 memory_queue: mp.Queue,
                 replay_in_queue: mp.Queue,
                 pipe: ParamPipe,
                 global_args: argparse.Namespace,
                 log_queue: torch.multiprocessing.Queue,
                 actor_params: Dict[str, Union[int, float]],
                 image_dir: str = None):
        r"""
        Continually steps the environment and pushes experiences to replay memory

        :param pipe: `ParamPipe` object for communication with the learner
        :param n_episodes: Number of episodes over which to train or test
        :param render_mode: How and whether to visibly render states
        :param memory_queue: The queue that experiences are put in for encoding
        :param replay_in_queue: The queue of encoded experiences for storage
        :param global_args: The namespace returned bye the global argument parser
        :param log_queue: Queue used to pass logging information to a handler
        :param actor_params: dictionary of parameters used to determine actor behavior
        :param image_dir: required if render_mode is not `False`
        """
        self.pipe = pipe
        self.replay_in_queue = replay_in_queue
        self.memory_queue = memory_queue
        self.log_queue = log_queue
        self.env = dispatch_make_env(global_args)
        self.policy = deepcopy(model)
        self.render_mode = render_mode
        self.n_episodes = n_episodes
        self.image_dir = image_dir

        self.test_mode = global_args.test
        self._parse_options(**actor_params)

        self._set_device()
        self.policy = self.policy.to(self.device)

        self.id = self.counter
        type(self).counter += 1

        self.logger = None
        self.episode = 0
        self.total_steps = 0
        self.history = []

        self.main_proc = mp.Process(target=self._main, name=f"Actor-{self.id}")

    def _parse_options(self, eps_decay: float, eps_end: float, eps_start: float, **kwargs) -> None:
        r"""
        Assign actor attributes

        :param eps_decay: Epsilon decay rate. See `Actor._epsilon`
        :param eps_end: Minimum epsilon
        :param eps_start: Maximum epsilon
        :return:
        """
        self.eps_decay = eps_decay
        self.eps_end = eps_end
        self.eps_start = eps_start

    def _set_device(self) -> None:
        r"""
        Set the device to use for inference. Use the second GPU if there is an extra one
        """
        if torch.cuda.device_count() > 1:
            self.device = 'cuda:1'
        else:
            self.device = 'cpu'

    def __del__(self):
        self._terminate()

    def _terminate(self) -> None:
        r"""
        Terminate and join running processes
        """
        if self.main_proc.pid is not None:
            self.main_proc.terminate()
            self.main_proc.join()

    def join(self) -> None:
        r"""
        Join the main actor process.
        """
        self.main_proc.join()

    def start(self) -> None:
        r"""
        Start worker processes and threads
        """
        self.main_proc.start()

    def _main(self):
        r"""
        Main worker process
        """
        encoder = Encoder(self.log_queue, self.replay_in_queue, self.memory_queue, self.id, daemon=True)
        encoder.start()

        self._set_num_threads(1)
        self.logger = get_logger_from_process(self.log_queue)
        self.logger.info(f"Actor-{self.id} started on device {self.device}")

        self._main_loop()
        self.memory_queue.put(None)  # signal encoder queue to terminate
        self.env.close()
        self._finish_rendering()
        encoder.join()
        self.logger.info(f"Actor-{self.id} done")

    def _main_loop(self):
        for self.episode in range(1, self.n_episodes + 1):
            self._run_episode()
            self.logger.debug(f"Actor-{self.id}, episode {self.episode} complete")
            self._log_episode()

    def _set_num_threads(self, n_threads: int) -> None:
        r"""
        Limits the number of threads that can be used by Pytorch if the device is set to cpu.

        :param n_threads: Maximum number of threads that may be used
        """
        if self.device == 'cpu':
            torch.set_num_threads(n_threads)

    def _update_params(self) -> None:
        r"""
        Request a parameter update from the learner
        """
        self.pipe.event.set()
        wait_event_not_set(self.pipe.event, timeout=None)
        params = self.pipe.conn_in.recv()
        self.policy.load_state_dict(params)
        self.logger.debug(f"Actor-{self.id} params updated")

    def _run_episode(self):
        r"""
        Run the agent for a full episode
        """
        obs = self.env.reset()
        state = get_state(obs, self.device)
        assert state.cpu.size() == (1, 4, 84, 84), self.logger.error(f"state is unexpected size: {state.cpu.size()}")
        total_reward = 0.0
        for steps in count():
            done, reward, state = self._run_step(state)
            total_reward += reward
            if self.total_steps % ACTOR_UPDATE_INTERVAL == 0:
                self._update_params()
            if done or steps >= MAX_STEPS_PER_EPISODE:
                break
        # noinspection PyUnboundLocalVariable
        self.history.append(HistoryElement(steps, total_reward))

    def _run_step(self, state: State) -> Tuple[bool, Union[int, float], State]:
        r"""
        Run the agent for a step. `state` is the previous environment state, used to determine the next action.

        :param state: `torch.Tensor`
        :return: (done, reward, next_state)
        """
        self.total_steps += 1

        action = self._select_action(state.cuda)
        del state.cuda

        self._dispatch_render()
        obs, reward, done, info = self.env.step(action)
        if not done:
            next_state = get_state(obs, self.device)
        else:
            next_state = State(None)

        self._end_step(action, done, next_state, reward, state)
        return done, reward, next_state

    def _end_step(self, action, done, next_state, reward, state):
        """
        Do whatever needs to be done after the step is complete
        """
        self.memory_queue.put(
            Transition(self.id, self.total_steps, state.cpu, action, next_state.cpu, reward, done)
        )
        if self.memory_queue.full():
            self.logger.debug(f'memory_queue FULL')

    def _select_action(self, state: torch.Tensor) -> int:
        r"""
        Select an action based on the current state and current exploration method

        :param state: Current environment state
        :return: action
        """
        if random.random() > self._epsilon:
            with torch.no_grad():
                return self.policy(state).max(1)[1].item()
        else:
            return random.randrange(N_ACTIONS)

    @property
    def _epsilon(self) -> float:
        r"""
        The probability of selecting a random action in epsilon-greedy exploration with episode decay.

            .. math::
                \epsilon_{end} + (\epsilon_{start} - \epsilon_{end}) * e^{- \frac{episode}{\epsilon_{decay}}}

        :return: probability
        """
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.episode / self.eps_decay)

    def _log_episode(self) -> None:
        r"""
        Log the results of the last episode
        """
        self.logger.info(f'Actor: {self.id:<3}\t'
                         f'Total steps: {self.total_steps:<9}\t'
                         f'Episode: {self.episode:<5}\t'
                         f'Reward: {int(self.history[-1].total_reward)}\t'
                         f'Steps: {self.history[-1].n_steps}')

    def _dispatch_render(self) -> None:
        r"""
        Render if `render_mode` is set
        """
        if self.render_mode:
            self.env.render(mode=self.render_mode, save_dir=self.image_dir)
            time.sleep(0.02)

    def _finish_rendering(self):
        r"""
        If `render_mode` set to 'png', convert stored images to a video.
        """
        # todo: test this functionality, and also that of storing the png images
        if self.render_mode == 'png':
            convert_images_to_video(image_dir=self.image_dir, save_dir=os.path.dirname(self.image_dir))
            shutil.rmtree(self.image_dir)


class ActorTest(Actor):
    """
    Not a true subclass of Actor. Runs blocking in a single process.
    """
    counter = 0

    # noinspection PyMissingConstructor
    def __init__(self,
                 model: nn.Module,
                 n_episodes: int,
                 render_mode: Union[str, bool],
                 global_args: argparse.Namespace,
                 actor_params: Dict[str, Union[int, float]],
                 logger: logging.Logger,
                 image_dir: str = None):
        r"""
        Continually steps the environment and pushes experiences to replay memory

        :param n_episodes: Number of episodes over which to train or test
        :param render_mode: How and whether to visibly render states
        :param global_args: The namespace returned bye the global argument parser
        :param logger: Main process logger
        :param actor_params: dictionary of parameters used to determine actor behavior
        :param image_dir: required if render_mode is not `False`
        """
        self.logger = logger
        self.env = dispatch_make_env(global_args)
        self.policy = deepcopy(model)
        self.render_mode = render_mode
        self.n_episodes = n_episodes
        self.image_dir = image_dir

        self.test_mode = global_args.test
        self._parse_options(**actor_params)

        self._set_device()
        self.policy = self.policy.to(self.device)

        self.id = self.counter
        type(self).counter += 1

        self.episode = 0
        self.total_steps = 0
        self.history = []
        self.memory = []

    def __del__(self):
        pass

    # noinspection PyMethodOverriding
    def _parse_options(self, eps_decay: float, eps_end: float, eps_start: float, save_transitions: bool,
                       **kwargs) -> None:
        r"""
        Assign actor attributes

        :param eps_decay: Epsilon decay rate. See `Actor._epsilon`
        :param eps_end: Minimum epsilon
        :param eps_start: Maximum epsilon
        :param save_transitions: If true, save transitions in  a pickle file
        :return:
        """
        self.eps_decay = eps_decay
        self.eps_end = eps_end
        self.eps_start = eps_start
        self.save_transitions = save_transitions

    def _main(self):
        raise NotImplementedError

    def _terminate(self):
        raise NotImplementedError

    def _update_params(self) -> None:
        pass

    def join(self) -> None:
        pass

    def start(self) -> None:
        self._main_loop()
        self.env.close()
        self._finish_rendering()
        self._save_transitions()

    def _save_transitions(self):
        if self.save_transitions:
            with open('memory.p', 'wb') as f:
                pickle.dump(self.memory, f)

    def _end_step(self, action, done, next_state, reward, state):
        if self.save_transitions:
            self.memory.append(
                (self.id, self.total_steps, state.cpu, action, next_state.cpu, reward, done)
            )


class Encoder(BaseWorker):
    def __init__(self, log_queue: mp.Queue, replay_in_queue: mp.Queue, memory_queue: mp.Queue, num: int, daemon=True):
        self.log_queue = log_queue
        self.replay_in_queue = replay_in_queue
        self.memory_queue = memory_queue

        self.proc = mp.Process(target=self._main, name=f"Encoder-{num}", daemon=daemon)

    def __del__(self):
        self._terminate()

    def _parse_options(self, **kwargs):
        pass

    def _set_device(self):
        pass

    def _terminate(self):
        self.proc.terminate()
        self.proc.join()

    def start(self):
        self.proc.start()

    def join(self):
        self.proc.join()

    def _main(self):
        r"""
        Encoder worker to be run alongside Actors
        """
        transition: Transition

        self.logger = get_logger_from_process(self.log_queue)
        self.logger.debug("Memory encoder process started")
        while True:
            transition = self.memory_queue.get()
            if transition is None:  # The actor is done
                break

            actor_id, step_number, state, action, next_state, reward, done = transition
            del transition

            if self.memory_queue.empty():
                self.logger.debug(f'memory_queue EMPTY')

            self._check_inputs(action, next_state, reward, state)
            next_state, state = self._process_states(next_state, state)

            self.replay_in_queue.put(
                self._get_transition(action, actor_id, done, next_state, reward, state, step_number)
            )

            if self.replay_in_queue.full():
                self.logger.debug(f'replay_in_queue FULL')

    @staticmethod
    def _process_states(next_state, state) -> Tuple[np.ndarray, np.ndarray]:
        """
        Squeeze and convert to numpy
        """
        state = state.squeeze().numpy()
        if next_state is not None:
            next_state = next_state.squeeze().numpy()
        return next_state, state

    def _check_inputs(self, action, next_state, reward, state):
        """
        Ensure inputs are the expected types
        """
        assert isinstance(state, torch.Tensor), self.logger.error(f"state must be a Tensor, not {type(state)}")
        assert isinstance(next_state, (torch.Tensor, type(None))), \
            self.logger.error(f"next_state must be a Tensor or None, not{type(next_state)}")
        assert isinstance(action, int), self.logger.error(f"action must be an integer, not {type(action)}")
        assert isinstance(reward, (int, float)), self.logger.error(f"reward must be a float, not {type(reward)}")

    def _get_transition(self, action, actor_id, done, next_state, reward, state, step_number) -> Transition:
        """
        Return a transition object. Override in a subclass to do work on any of the parameters before storage.
        """
        return Transition(actor_id, step_number, state, action, next_state, reward, done)


class EncoderCompress(Encoder):
    def _get_transition(self, action, actor_id, done, next_state, reward, state, step_number) -> Transition:
        """
        Compress states as PNG images before storage.
        """
        png_next_state, png_state = self._compress_states(next_state, state)
        return Transition(actor_id, step_number, png_state, action, png_next_state, reward, done)

    def _compress_states(self, next_state: np.ndarray, state: np.ndarray) -> Tuple[Union[io.BytesIO, None], io.BytesIO]:
        r"""
        Convert `state` and `next_state` into png image files

        :param next_state: a numpy array or None
        :param state: a numpy array
        :return: `(png_next_state, png_state)`
        """
        png_next_state = None
        if state.ndim == 2:
            png_state = io.BytesIO()
            png_next_state = io.BytesIO()
            Image.fromarray(state).save(png_state, format='png')
            if next_state is not None:
                Image.fromarray(next_state).save(png_next_state, format='png')
        else:
            png_state = self._encode_stacked_frames(state)
            if next_state is not None:
                png_next_state = self._encode_stacked_frames(next_state)
        return png_next_state, png_state

    @staticmethod
    def _encode_stacked_frames(state: np.ndarray) -> List[io.BytesIO]:
        r"""
        Return a list of png image files for each frame along the zeroth dimension of `state`

        :param state: a numpy array
        :return: a list of png files
        """
        result = []
        for frame in state:
            f = io.BytesIO()
            Image.fromarray(frame).save(f, format='png')
            result.append(f)
        return result


def wait_event_not_set(event: mp.Event, timeout: Union[None, float] = None):
    r"""
    Blocks until the event is *not* set. Raises TimeoutError if the timeout is reached.

    :param event: multiprocessing.Event to wait for
    :param timeout: timeout (s)
    """
    if timeout is None:
        timeout = math.inf

    start_time = time.time()
    elapsed = 0
    while event.is_set() and elapsed < timeout:
        time.sleep(0.01)
        elapsed = time.time() - start_time

    if elapsed >= timeout:
        raise TimeoutError
