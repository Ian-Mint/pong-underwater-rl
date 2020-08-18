#!/usr/bin/env python3
"""
Distributed reinforcement learning for gym-dynamic-pong. Requires a Linux operating system.

Memory and sampling pipeline:
Actor -> memory_queue
                    |
Encoder             +-> replay_in_queue
                                      |
Replay                                +-> memory
                                               |
Replay                                         +-> replay_out_queue
                                                                  |
Decoder                                                           +-> sample_queue
                                                                                 |
Learner                                                                          +-> _sample
"""
import abc
import argparse
import ctypes
import io
import logging
import math
import os
import random
import shutil
import sys
import threading
import time
from collections import namedtuple, OrderedDict
from copy import deepcopy
from itertools import count
from logging.handlers import QueueHandler
from multiprocessing.shared_memory import SharedMemory
from typing import Union, List, Dict, Tuple, Callable, Iterable

import gym
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms

sys.path.append(os.path.dirname(__file__))
if not ('linux' in sys.platform):
    raise Warning(f"{sys.platform} is not supported")

try:
    from .memory import *
    from .models import *
    from .wrappers import *
    from .utils import convert_images_to_video, distr_projection, get_args_status_string, get_logger
except ImportError:
    from memory import *
    from models import *
    from wrappers import *
    from utils import convert_images_to_video, distr_projection, get_args_status_string, get_logger

# Constants
MEMORY_BATCH_SIZE = 100
MAX_STEPS_PER_EPISODE = 50_000
N_ACTIONS = 3
ACTOR_UPDATE_INTERVAL = 1000
LOG_INTERVAL = 20  # number of episodes between logging
CHECKPOINT_INTERVAL = 1000  # number of batches between storing a checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

# Utility classes
Transition = namedtuple('Transition', ('actor_id', 'step_number', 'state', 'action', 'next_state', 'reward', 'done'))
HistoryElement = namedtuple('HistoryElement', ('n_steps', 'total_reward'))


class Memory:
    """
    A shared memory utility class
    """

    def __init__(self):
        n_bytes = 28224  # 84x84 numpy float64 array size

        self.array_dtype = 'uint8'
        self.array_shape = (4, 84, 84)

        self.actor_id = mp.Value(ctypes.c_int, 0, lock=False)
        self.step_number = mp.Value(ctypes.c_int, 0, lock=False)
        self.state = SharedMemory(create=True, size=n_bytes)
        self.action = mp.Value(ctypes.c_int, 0, lock=False)
        self.next_state = SharedMemory(create=True, size=n_bytes)
        self.reward = mp.Value(ctypes.c_int, 0, lock=False)
        self.done = mp.Value(ctypes.c_bool, 0, lock=False)
        self.lock = mp.Lock()

    def update(self, transition: Transition):
        """
        Store `transition` in shared memory

        :param transition: a `Transition`
        """
        with self.lock:
            self.actor_id.value = transition.actor_id
            self.step_number.value = transition.step_number
            self._copy_array_into_buffer(transition.state, self.state.buf)
            self.action.value = transition.action
            if not transition.done:
                self._copy_array_into_buffer(transition.next_state, self.next_state.buf)
            self.done.value = transition.done

    def get_values(self):
        """
        Get a clean transition from the shared memory objects

        :return: Transition
        """
        with self.lock:
            state = self._get_array_from_buffer(self.state.buf)
            done = self.done.value
            if done:
                next_state = None
            else:
                next_state = self._get_array_from_buffer(self.next_state.buf)
            return Transition(self.actor_id.value, self.step_number.value, state, self.action.value, next_state,
                              self.reward.value, done)

    def _get_array_from_buffer(self, buffer: memoryview) -> np.ndarray:
        """
        Copy array from shared memory

        :param buffer: `SharedMemory` buffer
        :return: a numpy array
        """
        array = np.ndarray(self.array_shape, dtype=self.array_dtype, buffer=buffer)
        return array.copy()

    def _copy_array_into_buffer(self, array: np.ndarray, buffer: memoryview) -> None:
        """
        Copy array into shared memory

        :param array: numpy array
        :param buffer: `SharedMemory` buffer
        """
        assert array.shape == self.array_shape, f"shape {array.shape} != {self.array_shape}"
        assert array.dtype == self.array_dtype, f"dtype {array.dtype} != {self.array_dtype}"
        s = np.ndarray(array.shape, dtype=array.dtype, buffer=buffer)
        s[:] = array[:]


class ProcessedBatch:
    def __init__(self,
                 actions: torch.Tensor,
                 rewards: torch.Tensor,
                 states: torch.Tensor,
                 non_final_mask: torch.Tensor,
                 non_final_next_states: torch.Tensor,
                 idxs: None,
                 weights: None):
        """
        For storing training batches and pushing them to the specified device

        :param actions: Action batch
        :param rewards: Rewards batch
        :param states: State batch
        :param non_final_mask: Boolean mask where 1 shows non-final states
        :param non_final_next_states: Masked next states
        :param idxs: Priority index of each state in batch
        :param weights: Priority weight of each state in batch
        """
        if (idxs is not None) or (weights is not None):
            raise NotImplementedError("prioritized replay not yet implemented")

        self.actions = actions
        self.rewards = rewards
        self.states = states
        self.non_final_mask = non_final_mask
        self.non_final_next_states = non_final_next_states
        self.idxs = idxs
        self.weights = weights

    def to(self, device, non_blocking=True):
        """
        Push all tensors to specified device

        :param device: Device to push to e.g. cuda, cpu
        :param non_blocking: If True, push to device asynchronously
        :return: Updated ProcessedBatch
        """
        self.actions = self.actions.to(device, non_blocking=non_blocking)
        self.rewards = self.rewards.to(device, non_blocking=non_blocking)
        self.states = self.states.to(device, non_blocking=non_blocking)
        self.non_final_mask = self.non_final_mask.to(device, non_blocking=non_blocking)
        self.non_final_next_states = self.non_final_next_states.to(device, non_blocking=non_blocking)
        return self


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


class ParamPipe:
    def __init__(self):
        """
        Creates and holds an event object and a multiprocessing connection pair.

        Utility class used for communicating between learner and actors.
        """
        self.event = torch.multiprocessing.Event()
        self.conn_in, self.conn_out = mp.Pipe(duplex=False)


def get_logger_from_process(log_queue: mp.Queue) -> logging.Logger:
    """
    To be called from new processes to generate a handle to the root logger.

    :param log_queue: Queue object used to store log messages
    :return: a Logger
    """
    logger = logging.getLogger()
    logger.addHandler(QueueHandler(log_queue))
    logger.setLevel(logging.DEBUG)
    return logger


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


# Main classes
class Worker(abc.ABC):
    """
    Base class for multiprocessing worker classes
    """

    @abc.abstractmethod
    def start(self): ...

    @abc.abstractmethod
    def _main(self): ...

    @abc.abstractmethod
    def _parse_options(self, **kwargs): ...

    @abc.abstractmethod
    def _set_device(self): ...

    @abc.abstractmethod
    def _terminate(self): ...

    @abc.abstractmethod
    def __del__(self): ...


class Learner(Worker):
    """
           +-------+                  +------------+
           |Learner|----------------->|Param Server|
           +-------+                  +------------+
             ^   ^                   ____/      \____
          __/     \___              /                \
        /             \            v                  v
    +-------+     +-------+   +----------+      +----------+
    |Decoder| ... |Decoder|   |Param Pipe| ...  |Param Pipe|
    +-------+     +-------+   +----------+      +----------+
    """

    def __init__(self,
                 optimizer: Callable,
                 model,
                 replay_out_queue: torch.multiprocessing.Queue,
                 sample_queue: torch.multiprocessing.Queue,
                 pipes: List[ParamPipe], checkpoint_path: str,
                 log_queue: torch.multiprocessing.Queue,
                 learning_params: Dict[str, Union[float, int]],
                 n_decoders: int = 6):
        """
        In two separate processes, decodes sampled data and runs training.

        :param n_decoders:
        :param optimizer: The selected type of optimizer
        :param model: The initialized model object to be copied into the learner
        :param replay_out_queue: _sample batches are pulled from this queue for decoding
        :param sample_queue: decoded batches are put on this queue for training
        :param pipes: list of `ParamPipe` objects for communicating with Actors
        :param checkpoint_path: Checkpoint save path
        :param log_queue: Queue object to be pushed to the log handler for the learner process
        :param learning_params: Parameters to control learning
        """
        self.replay_out_queue = replay_out_queue
        self.sample_queue = sample_queue
        self.pipes = pipes
        self.log_queue = log_queue

        self.checkpoint_path = checkpoint_path
        self._parse_options(**learning_params)

        self.n_decoder_processes = n_decoders
        self._set_device()

        self.policy = deepcopy(model).to(self.device)
        self.target = deepcopy(model).to(self.device)
        self.optimizer = optimizer(self.policy.parameters(), lr=self.learning_rate)

        self.params = None
        self.params_lock = None
        self.policy_lock = None
        self.logger = None
        self.loss = None
        self.epoch = 0

        self.main_proc = mp.Process(target=self._main, name="Learner")

    def _parse_options(self, batch_size: int, gamma: float, learning_rate: float, **kwargs) -> None:
        """
        Parse training options

        :param batch_size: Used in calculations, does not control batch size
        :param gamma: The reward decay rate
        :param learning_rate: The network learning rate
        """
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate

    def _set_device(self) -> None:
        """
        If we have a GPU, set the device to cuda, otherwise, set it to cpu
        """
        if torch.cuda.device_count() != 0:
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

    def __del__(self):
        self._terminate()

    def _terminate(self) -> None:
        """
        _terminate and join the main process
        """
        self.main_proc.terminate()
        self.main_proc.join()

    def start(self) -> None:
        """
        Start all threads and processes
        """
        # A started process has a `__weakref__` attribute that is not picklable. So, a started process cannot be passed
        # by context to another process. In this case, only one of the processes can be stored in `self`, and the other
        # process must be started before the `self` process.
        for i in range(self.n_decoder_processes):
            decoder = Decoder(self.log_queue, self.replay_out_queue, self.sample_queue, i, daemon=True)
            decoder.start()

        self.main_proc.start()
        del self.policy, self.target  # These have already been copied into the child process

    def _main(self) -> None:
        """
        The main worker process
        """
        self.logger = get_logger_from_process(self.log_queue)
        self.logger.info(f"Learner started on device {self.device}")

        self.policy_lock = threading.Lock()
        self.params_lock = threading.Lock()

        # TODO: put this stuff in a new process instead of threads so that we get 100% CPU usage on the learner
        param_update_thread = threading.Thread(target=self._copy_params, name='UpdateParams', daemon=True)
        param_update_thread.start()
        for n, p in enumerate(self.pipes):
            t = threading.Thread(target=self._send_params, args=(p,), name=f'SendParams-{n}', daemon=True)
            t.start()

        self._optimizer_loop()

    def _optimizer_loop(self) -> None:
        """
        Main training loop
        """
        for self.epoch in count(1):
            self._optimize_model()
            self._update_target_net()
            if self.epoch % CHECKPOINT_INTERVAL:
                self._save_checkpoint()

    def _copy_params(self) -> None:
        """
        Update the pipe every 2.5 seconds. Keep a lock to the pipe while it is being updated.
        """
        while True:
            with self.params_lock:
                with self.policy_lock:
                    self._policy_state_dict_to_params()
            time.sleep(2.5)

    def _policy_state_dict_to_params(self):
        cuda_state_dict = self.policy.state_dict()
        self.params = OrderedDict()
        for k, v in cuda_state_dict.items():
            self.params[k] = v.to('cpu')

    def _send_params(self, pipe: ParamPipe) -> None:
        """
        Thread to send params through `pipe`

        :param pipe: ParamPipe object associated with an actor
        """
        while self.params is None:
            time.sleep(1)

        while True:
            if pipe.event.is_set():
                with self.params_lock:
                    pipe.conn_out.send(self.params)
                pipe.event.clear()
            else:
                time.sleep(0.01)

    def _optimize_model(self):
        """
        Run a batch through optimization
        """
        batch = self._sample()

        state_action_values = self._forward_policy(batch.actions, batch.states)
        next_state_values = self._forward_target(batch.non_final_mask, batch.non_final_next_states)
        expected_state_action_values = (next_state_values * self.gamma) + batch.rewards.float()

        loss = self._get_loss(state_action_values, expected_state_action_values)
        self.logger.debug(f"loss norm: {loss.norm()}")
        self._step_optimizer(loss)

    def _sample(self) -> ProcessedBatch:
        """
        Pull a batch from the sample queue and push to `device`
        :return: ProcessedBatch object on `device`
        """
        processed_batch = self.sample_queue.get()
        if self.sample_queue.empty():
            self.logger.debug(f'sample_queue EMPTY')
        return processed_batch.to(self.device)

    def _forward_policy(self, action_batch: torch.Tensor, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Return :math:`Q_{\pi}(a,s)` as a tensor for all actions and states in `action_batch` and `state_batch`

        :param action_batch: Batch of actions
        :param state_batch: Batch of states
        :return: The Q-values of all action-state pairs
        """
        return self.policy(state_batch).gather(1, action_batch)

    def _forward_target(self, non_final_mask: torch.Tensor, non_final_next_states: torch.Tensor) -> torch.Tensor:
        """
        Return :math:`max_{a'} Q_{\pi'}(a',s')` as a tensor for all states in `non_final_next_states`

        :param non_final_mask: Boolean tensor of all states that are not final
        :param non_final_next_states: All next states that are not final
        :return: The maximum Q-value for each non-final next state
        """
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target(non_final_next_states).max(1)[0].detach()
        return next_state_values

    def _step_optimizer(self, loss: torch.Tensor) -> None:
        """
        Given the loss, run backpropagation and update parameters

        :param loss: Batch loss
        """
        self.optimizer.zero_grad()
        loss.backward()

        with self.policy_lock:
            self.optimizer.step()

    @staticmethod
    def _get_loss(state_action_values: torch.Tensor, expected_state_action_values: torch.Tensor) -> torch.Tensor:
        """
        Get L1 loss

        :param state_action_values: Q-values given the chosen actions
        :param expected_state_action_values: Q-values of the best actions
        :return: L1 loss
        """
        return F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    def _update_target_net(self) -> None:
        """
        Copy policy net parameters onto the target net
        """
        self.target.load_state_dict(self.policy.state_dict())

    def _save_checkpoint(self) -> None:
        """
        Save the current state dictionary of `Learner.policy` at `Learner.checkpoint_path`
        """
        torch.save(
            {'policy_state': self.policy.state_dict(),
             'optimizer_state': self.optimizer.state_dict(),
             'samples_processed': self.epoch * self.batch_size,
             'epoch': self.epoch},
            os.path.join(self.checkpoint_path))


class Decoder(Worker):
    def __init__(self, log_queue: mp.Queue, replay_out_queue: mp.Queue, sample_queue: mp.Queue, num: int, daemon=True):
        self.log_queue = log_queue
        self.replay_out_queue = replay_out_queue
        self.sample_queue = sample_queue

        self.proc = mp.Process(target=self._main, name=f"MemoryDecoder-{num}", daemon=daemon)

    def start(self):
        self.proc.start()

    def _parse_options(self, **kwargs):
        pass

    def _set_device(self):
        pass

    def _terminate(self):
        self.proc.terminate()
        self.proc.join()

    def __del__(self):
        self._terminate()

    def _main(self, compress: bool = False) -> None:
        """
        Decoder worker to be run alongside Learner. To save GPU memory, we leave it to the Learner to push tensors to
        GPU.

        :param compress: If true, compress frames as PNG images. Saves ~50% memory, but will require multiple workers to
                         feed the learner quickly enough.
        """
        transition: Transition

        self.logger = get_logger_from_process(self.log_queue)
        self.logger.debug("Memory decoder process started")

        while True:
            batch = self.replay_out_queue.get()
            if self.replay_out_queue.empty():
                self.logger.debug(f'replay_out_queue EMPTY')

            decoded_batch = []
            for transition in batch:
                decoded_batch.append(self._decode_transition(transition))

            batch, actions, rewards = self._process_transitions(decoded_batch)
            non_final_mask, non_final_next_states = self._mask_non_final(batch)
            action_batch, reward_batch, state_batch = self._concatenate_batches(actions, rewards, batch.state)
            processed_batch = ProcessedBatch(action_batch, reward_batch, state_batch,
                                             non_final_mask, non_final_next_states,
                                             idxs=None, weights=None)

            self.sample_queue.put(processed_batch)
            if self.sample_queue.full():
                self.logger.debug(f'sample_queue FULL')

    def _decode_transition(self, transition: Transition) -> Transition:
        actor_id, step_number, state, action, next_state, reward, done = transition
        next_state, state = self.states_to_tensor(next_state, state)
        return Transition(actor_id, step_number, state, action, next_state, reward, done)

    def states_to_tensor(self, next_state: Union[np.ndarray, None], state: np.ndarray) \
            -> Tuple[Union[torch.Tensor, None], torch.Tensor]:
        """
        Converts `next_state` and `state` from numpy arrays to `torch.Tensor`s.

        :param next_state: numpy array or None
        :param state: numpy array
        :return: `(next_state, state)`
        """
        state = self.to_tensor(state)
        next_state = self.to_tensor(next_state)
        return next_state, state

    @staticmethod
    def to_tensor(state: Union[np.ndarray, None]) -> torch.Tensor:
        """
        Converts a numpy array to a pytorch tensor and unsqueezes the zeroth dimension.

        :param state: Numpy array or None
        :return: `torch.Tensor`
        """
        if state is not None:
            state = torch.from_numpy(state).to('cpu')
            state = state.unsqueeze(0)
        return state

    @staticmethod
    def _concatenate_batches(*args: Union[List[torch.Tensor], Tuple[torch.Tensor]]) -> Tuple[torch.Tensor, ...]:
        """
        For each argument, concatenate all of its elements into a single `torch.Tensor`. Return a tuple of these
        concatenated tensors.

        :param args: List or tuple of `torch.Tensor`
        """
        result = (torch.cat(a).to('cpu') for a in args)
        return tuple(result)

    @staticmethod
    def _mask_non_final(batch: Transition) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mask None values in `batch.next_state`

        :param batch: `Transition` containing a list of tensors for each element
        :return: (mask, masked_next_states)
        """
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device='cpu', dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to('cpu')
        return non_final_mask, non_final_next_states

    @staticmethod
    def _process_transitions(transitions: List[Transition]) -> \
            Tuple[Transition, Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """
        Convert a list of transitions into:
            - Transition object containing a list of each type of element
            - a tuple of action tensors
            - a tuple of rewards tensors

        :param transitions: A list of `Transition`s
        :return: `(batch, actions, rewards)`
        """
        batch = Transition(*zip(*transitions))
        actions = tuple((map(lambda a: torch.tensor([[a]], device='cpu'), batch.action)))
        rewards = tuple((map(lambda r: torch.tensor([r], device='cpu'), batch.reward)))
        return batch, actions, rewards


class DecoderCompress(Decoder):
    def _decode_transition(self, transition: Transition) -> Transition:
        actor_id, step_number, png_state, action, png_next_state, reward, done = transition
        next_state, state = self._decompress_states(png_next_state, png_state)
        return Transition(actor_id, step_number, state, action, next_state, reward, done)

    def _decompress_states(self,
                           png_next_state: Union[List[Union[io.BytesIO, None]], Union[io.BytesIO, None]],
                           png_state: io.BytesIO) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Convert png image states stored as a `BytesIO` file to a `torch.Tensor`. On png_next_state, performs checking in
        case the state is `None`.

        :param png_next_state: png image stored in `BytesIO` or `None`
        :param png_state: png image stored in `BytesIO`
        :return: `(next_state, state)`
        """
        transform = transforms.ToTensor()
        next_state = None
        if isinstance(png_state, list):
            state = self._decode_stacked_frames(png_state)
            if png_next_state is not None:
                next_state = self._decode_stacked_frames(png_next_state)
        else:
            state = transform(Image.open(png_state)).to('cpu')
            if png_next_state is not None:
                next_state = transform(Image.open(png_next_state)).to('cpu')
        return next_state, state

    @staticmethod
    def _decode_stacked_frames(png_state: List[io.BytesIO]) -> torch.Tensor:
        """
        Convert a list of png files to a `torch.Tensor`, with a zeroth dimension of 1, and first dimension equal to the
        `len(png_state)`.

        :param png_state: A list of png file objects
        :return: A `torch.Tensor` with `size((1, len(png_state), height, width))`
        """
        transform = transforms.ToTensor()
        result = []
        for f in png_state:
            frame = transform(Image.open(f))
            result.append(frame.squeeze())
        return torch.stack(result).unsqueeze(0).to('cpu')


class Actor(Worker):
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
                 memory_queue: torch.multiprocessing.Queue,
                 replay_in_queue: torch.multiprocessing.Queue,
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

    def _encoder_start(self) -> mp.Process:
        """
        Creates and starts the encoder process

        :return: encoder process handle
        """
        encoder_proc = mp.Process(target=self._encoder_worker, name=f"Encoder-{self.id}",
                                  kwargs=dict(compress=False), daemon=True)
        encoder_proc.start()
        return encoder_proc

    def _main(self):
        r"""
        Main worker process
        """
        encoder_proc = self._encoder_start()
        self._set_num_threads(1)
        self.logger = get_logger_from_process(self.log_queue)
        self.logger.info(f"Actor-{self.id} started on device {self.device}")

        for self.episode in range(1, self.n_episodes + 1):
            self._run_episode()
            self.logger.debug(f"Actor-{self.id}, episode {self.episode} complete")
            self._log_episode()
        self.memory_queue.put(None)  # signal encoder queue to terminate
        self.env.close()
        self._finish_rendering()
        encoder_proc.join()
        self.logger.info(f"Actor-{self.id} done")

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
        if not self.test_mode:
            self.memory_queue.put(
                Transition(self.id, self.total_steps, state.cpu, action, next_state.cpu, reward, done)
            )
            if self.memory_queue.full():
                self.logger.debug(f'memory_queue FULL')
        return done, reward, next_state

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

    def _encoder_worker(self, compress=False):
        # todo: separate out an Encoder class. There is no overlap between learner and encoder methods
        r"""
        Encoder worker to be run alongside Actors

        :param compress: if True, compress states as PNG images
        """
        transition: Transition

        self.logger = get_logger_from_process(self.log_queue)
        self.logger.debug("Memory encoder process started")
        while True:
            transition = self.memory_queue.get()
            if transition is None:
                break
            else:
                actor_id, step_number, state, action, next_state, reward, done = transition
                del transition

            if self.memory_queue.empty():
                self.logger.debug(f'memory_queue EMPTY')
            assert isinstance(state, torch.Tensor), self.logger.error(f"state must be a Tensor, not {type(state)}")
            assert isinstance(next_state, (torch.Tensor, type(None))), \
                self.logger.error(f"next_state must be a Tensor or None, not{type(next_state)}")
            assert isinstance(action, int), self.logger.error(f"action must be an integer, not {type(action)}")
            assert isinstance(reward, (int, float)), self.logger.error(f"reward must be a float, not {type(reward)}")

            state = state.squeeze().numpy()
            if next_state is not None:
                next_state = next_state.squeeze().numpy()

            # todo: put compression in a subclass
            if compress:
                png_next_state, png_state = self._compress_states(next_state, state)
                self.replay_in_queue.put(
                    Transition(actor_id, step_number, png_state, action, png_next_state, reward, done)
                )
            else:
                self.replay_in_queue.put(
                    Transition(actor_id, step_number, state, action, next_state, reward, done)
                )
            if self.replay_in_queue.full():
                self.logger.debug(f'replay_in_queue FULL')

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


class Replay(Worker):
    r"""
                     +-------------+
    process          |    _main    |
                     +-------------+
                     /            \
                    /              \
    subprocess  _push_worker   _sample_worker
    """

    def __init__(self, replay_in_queue, replay_out_queue, log_queue, params: Dict[str, Union[int, float]],
                 mode='default'):
        r"""
        Stores and samples the replay memory.

        :param replay_in_queue: The queue used to retrieve samples
        :param replay_out_queue: The queue used to push samples
        :param params: parameters controlling the replay memory behavior (length, etc.)
        :param mode: {'default', 'episodic'}
        """
        # todo: implement episodic replay for use with LSTM
        if mode != 'default':
            raise NotImplementedError("Only default mode is currently implemented")

        assert params['initial_memory'] <= params['memory_size'], \
            "Initial replay memory set lower than total replay memory"

        self.initial_memory = params['initial_memory']
        self.batch_size = params['batch_size']

        self.replay_in_queue = replay_in_queue
        self.replay_out_queue = replay_out_queue
        self.log_queue = log_queue
        self.mode = mode

        self.buffer_in = []
        self.memory_maxlen = params['memory_size']
        self.memory = tuple(Memory() for _ in range(self.memory_maxlen))
        self.memory_length = mp.Value('i', 0)
        self.sample_count = 0

        self.logger = None
        self.memory_full_event = mp.Event()
        self._main_proc = mp.Process(target=self._main, name="Replay")

    def __del__(self):
        self._terminate()

    def _parse_options(self, **kwargs):
        raise NotImplementedError

    def _set_device(self):
        raise NotImplementedError

    def _terminate(self):
        r"""
        Main thread
        """
        self._main_proc.terminate()
        self._main_proc.join()

    def start(self):
        r"""
        Main thread
        """
        push_proc = mp.Process(target=self._push_worker, daemon=True, name="Push")
        push_proc.start()
        self._main_proc.start()

    def _main(self):
        r"""
        Launch push thread and run push worker in the main thread.
        """
        self.logger = get_logger_from_process(self.log_queue)
        self.logger.debug("Replay process started")

        self._sample_worker()  # run _sample worker in the main process

    def _push_worker(self) -> None:
        r"""
        Pushes from replay_in_queue to `memory`
        """
        self.logger = get_logger_from_process(self.log_queue)
        self.logger.debug("Replay memory push worker started")

        while True:
            sample = self.replay_in_queue.get()
            if self.replay_in_queue.empty():
                self.logger.debug(f'replay_in_queue EMPTY')

            self.buffer_in.append(sample)
            if len(self.buffer_in) >= self.initial_memory // 100:
                for transition in self.buffer_in:
                    memory = self.memory[self.sample_count % self.memory_maxlen]
                    memory.update(transition)

                    self.sample_count += 1
                    if not self.memory_full_event.is_set() and self.sample_count >= self.initial_memory:
                        self.memory_full_event.set()
                self.buffer_in = []

    def _sample_worker(self) -> None:
        r"""
        Generates samples from memory of length `batch_size` and pushes to `replay_out_queue`
        """
        self.logger.debug("Replay memory _sample worker started")
        self._wait_for_full_memory()

        while True:
            samples = random.choices(self.memory, k=self.batch_size)

            batch = [s.get_values() for s in samples]
            self.replay_out_queue.put(batch)
            if self.replay_out_queue.full():
                self.logger.debug(f'replay_out_queue FULL')

    def _wait_for_full_memory(self) -> None:
        """
        Blocks until the memory length surpasses `initial_memory`
        """
        while not self.memory_full_event.is_set():
            time.sleep(1)


def display_state(state: torch.Tensor) -> None:
    r"""
    Displays the passed state using matplotlib

    :param state: torch.Tensor
    """
    np_state = state.numpy().squeeze()
    fig, axs = plt.subplots(1, len(np_state), figsize=(20, 5))
    for img, ax in zip(np_state, axs):
        ax.imshow(img, cmap='gray')
    fig.show()


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


def initialize_model(architecture: str) -> nn.Module:
    r"""
    Return model based on the chosen architecture.

    :param architecture: model selector string
    :return: Selected model object
    """
    model_lookup = {'dqn': DQN,
                    'soft_dqn': softDQN,
                    'dueling_dqn': DuelingDQN,
                    'lstm': DRQN,
                    'distributional_dqn': DistributionalDQN}
    return model_lookup[architecture](n_actions=N_ACTIONS)  # Allow users of the model to put it on the desired device


def get_parser() -> argparse.ArgumentParser:
    """
    Generate command line argument parser

    :return: parser
    """
    parser = argparse.ArgumentParser(description='Dynamic Pong RL')

    '''environment args'''
    env_args = parser.add_argument_group('Environment', "Environment controls")
    env_args.add_argument('--width', default=160, type=int,
                          help='canvas width (default: 160)')
    env_args.add_argument('--height', default=160, type=int,
                          help='canvas height (default: 160)')
    env_args.add_argument('--ball', default=1.0, type=float,
                          help='ball speed (default: 1.0)')
    env_args.add_argument('--ball-size', dest='ball_size', default=2.0, type=float,
                          help='ball size (default: 2.0)')
    env_args.add_argument('--ball-volume', dest='ball_volume', action='store_true', default=False,
                          help='If set, the ball interacts as if it has volume')
    env_args.add_argument('--snell', default=1.0, type=float,
                          help='snell speed (default: 1.0)')
    env_args.add_argument('--no-refraction', dest='no_refraction', default=False, action='store_true',
                          help='set to disable refraction')
    env_args.add_argument('--uniform-speed', dest='uniform_speed', default=False, action='store_true',
                          help='set to disable a different ball speed in the Snell layer')
    env_args.add_argument('--snell-width', dest='snell_width', default=80.0, type=float,
                          help='snell speed (default: 80.0)')
    env_args.add_argument('--snell-change', dest='snell_change', default=0, type=float,
                          help='Standard deviation of the speed change per step (default: 0)')
    env_args.add_argument('--snell-visible', dest='snell_visible', default='none', type=str,
                          choices=['human', 'machine', 'none'],
                          help="Determine whether snell is visible to when rendering ('render') or to the agent and "
                               "when rendering ('machine')")
    env_args.add_argument('--paddle-speed', default=1.0, type=float,
                          help='paddle speed (default: 1.0)')
    env_args.add_argument('--paddle-angle', default=70, type=float,
                          help='Maximum angle the ball can leave the paddle (default: 70deg)')
    env_args.add_argument('--paddle-length', default=20, type=float,
                          help='paddle length (default: 20)')
    env_args.add_argument('--update-prob', dest='update_prob', default=0.4, type=float,
                          help='Probability that the opponent moves in the direction of the ball (default: 0.4)')
    env_args.add_argument('--state', default='binary', type=str, choices=['binary', 'color'],
                          help='state representation (default: binary)')

    '''RL args'''
    rl_args = parser.add_argument_group("Model", "Reinforcement learning model parameters")
    rl_args.add_argument('--learning-rate', default=1e-4, type=float,
                         help='learning rate (default: 1e-4)')
    rl_args.add_argument('--network', default='dqn',
                         choices=['dqn', 'soft_dqn', 'dueling_dqn', 'resnet18', 'resnet10', 'resnet12',
                                  'resnet14', 'lstm', 'distribution_dqn'],
                         help='choose a network architecture (default: dqn)')
    rl_args.add_argument('--double', default=False, action='store_true',
                         help='switch for double dqn (default: False)')
    rl_args.add_argument('--pretrain', default=False, action='store_true',
                         help='switch for pretrained network (default: False)')
    rl_args.add_argument('--test', default=False, action='store_true',
                         help='Run the model without training')
    rl_args.add_argument('--render', default=False, type=str, choices=['human', 'png'],
                         help="Rendering mode. Omit if no rendering is desired.")
    rl_args.add_argument('--epsdecay', default=1000, type=int,
                         help="_epsilon decay (default: 1000)")
    rl_args.add_argument('--steps-decay', dest='steps_decay', default=False, action='store_true',
                         help="switch to use default step decay")
    rl_args.add_argument('--episodes', dest='episodes', default=4000, type=int,
                         help='Number of episodes to train for (default: 4000)')
    rl_args.add_argument('--replay', default=100_000, type=int,
                         help="change the replay mem size (default: 100,000)")
    rl_args.add_argument('--priority', default=False, action='store_true',
                         help='switch for prioritized replay (default: False)')
    rl_args.add_argument('--rank-based', dest='rank_based', default=False, action='store_true',
                         help='switch for rank-based prioritized replay (omit if proportional)')
    rl_args.add_argument('--batch-size', dest='batch_size', default=512, type=int,
                         help="network training batch size or sequence length for recurrent networks")
    rl_args.add_argument('--compress', dest='compress_state', action='store_true', default=False,
                         help="If set, store states compressed as png images. Add one CPU if set")
    rl_args.add_argument('--actors', dest='n_actors', type=int, default=1,
                         help="Number of actors to use. 3 + n_actors CPUs required")

    '''resume args'''
    resume_args = parser.add_argument_group("Resume", "Store experiments / Resume training")
    resume_args.add_argument('--resume', dest='resume', action='store_true',
                             help='Resume training switch. (omit to start from scratch)')
    resume_args.add_argument('--checkpoint', default='model.torch',
                             help='Checkpoint to load if resuming (default: model.torch)')
    resume_args.add_argument('--history', default='history.p',
                             help='History to load if resuming (default: history.p)')
    resume_args.add_argument('--store-dir', dest='store_dir',
                             default=os.path.join('..', 'experiments', time.strftime("%Y-%m-%d %H.%M.%S")),
                             help='Path to directory to store experiment results (default: ./experiments/<timestamp>/')

    '''debug args'''
    debug_args = parser.add_argument_group("Debug")
    debug_args.add_argument('--debug', action='store_true', help='Debug mode')
    return parser


def create_storage_dir(directory: str) -> None:
    r"""
    Create directory `dir` if it does not exist
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def main() -> None:
    """
    Code here is called in the main process, but not by subprocesses.
    """
    parser = get_parser()
    args = parser.parse_args()
    create_storage_dir(args.store_dir)

    learning_params = {
        'batch_size': args.batch_size,
        'gamma': 0.99,
        'learning_rate': args.learning_rate,
        'prioritized': args.priority,
        'double': args.double,
        'architecture': args.network,
    }

    # todo: subclass actor to implement steps_decay
    if args.steps_decay:
        raise NotImplementedError("steps_decay is not yet implemented")

    actor_params = {
        'test_mode': args.test,
        'architecture': args.network,
        'steps_decay': args.steps_decay,
        'eps_decay': args.epsdecay,
        'eps_end': 0.02,
        'eps_start': 1,
    }

    replay_params = {
        'memory_size': args.replay,
        'initial_memory': args.replay,  # Wait for the memory buffer to fill before training
        'batch_size': args.batch_size
    }

    logger, log_queue = get_logger(args.store_dir)
    logger.info(get_args_status_string(parser, args))
    logger.info(f'Device: {DEVICE}')

    # Get shared objects
    model = initialize_model(args.network)
    memory_queue, replay_in_queue, replay_out_queue, sample_queue, pipes = get_communication_objects(args.n_actors)

    # Create subprocesses
    actors = []
    for p in pipes:
        a = Actor(model=model,
                  n_episodes=args.episodes,
                  render_mode=args.render,
                  memory_queue=memory_queue,
                  replay_in_queue=replay_in_queue,
                  pipe=p,
                  global_args=args,
                  log_queue=log_queue,
                  actor_params=actor_params)
        actors.append(a)

    learner = Learner(optimizer=optim.Adam, model=model, replay_out_queue=replay_out_queue, sample_queue=sample_queue,
                      pipes=pipes, checkpoint_path=os.path.join(args.store_dir, 'dqn.torch'), log_queue=log_queue,
                      learning_params=learning_params)
    replay = Replay(replay_in_queue, replay_out_queue, log_queue, replay_params)

    # Start subprocesses
    run_all(actors, 'start')
    replay.start()
    learner.start()

    try:
        # Join subprocess. actor is the only one that is not infinite.
        run_all(actors, 'join')
        logger.info("All actors finished")
    except KeyboardInterrupt:
        run_all(actors, '__del__')
    finally:
        del replay
        del learner


def run_all(actors: Iterable[Actor], method: str, *args, **kwargs) -> None:
    r"""
    Executes `method` for all actors in `actors`. Equivalent to
    ```
    for a in actors:
        a.method(*args, **kwargs)
    ```

    :param actors: Iterable of actor objects
    :param method: Actor.method to execute
    :param args:
    :param kwargs:
    """
    for a in actors:
        a.__getattribute__(method)(*args, **kwargs)


def get_communication_objects(n_pipes: int) -> Tuple[mp.Queue, mp.Queue, mp.Queue, mp.Queue, List[ParamPipe]]:
    r"""
    Return the various queues and pipes used to communicate between processes

    :param n_pipes: Number of ParamPipes. Should equal the number of actors.
    :return: (memory_queue, replay_in_queue, replay_out_queue, sample_queue, pipes)
    """
    memory_queue = mp.Queue(maxsize=1_000)
    replay_in_queue = mp.Queue(maxsize=1_000)
    replay_out_queue = mp.Queue(maxsize=100)
    sample_queue = mp.Queue(maxsize=20)

    pipes = [ParamPipe() for _ in range(n_pipes)]
    return memory_queue, replay_in_queue, replay_out_queue, sample_queue, pipes


if __name__ == '__main__':
    mp.set_start_method('forkserver')  # CUDA is incompatible with 'fork'
    main()
