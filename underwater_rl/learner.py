import io
import os
import threading
import time
from collections import OrderedDict
from copy import deepcopy
from itertools import count
from multiprocessing.managers import DictProxy
from typing import Callable, List, Dict, Union, Tuple
import sys

import numpy as np
import torch
from PIL import Image
from torch import multiprocessing as mp
from torch.nn import functional as F
from torchvision import transforms

try:
    import underwater_rl
except ImportError:
    sys.path.append(os.path.abspath(os.path.pardir))
from underwater_rl.common import BaseWorker, Transition
from underwater_rl.utils import get_logger_from_process, get_tid

CHECKPOINT_INTERVAL = 100  # number of batches between storing a checkpoint
TARGET_UPDATE_INTERVAL = 100  # number of batches between updating the target network


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


class Decoder(BaseWorker):
    """
    Decoder worker. One or more may be run alongside the learner to process sample batches.

    The bottleneck seems to be with `replay_out_queue`. Currently, two decoders can handle the output.
    """

    def __init__(self, log_queue: mp.Queue, replay_out_queue: mp.Queue, sample_queue: mp.Queue, num: int, daemon=True):
        self.log_queue = log_queue
        self.replay_out_queue = replay_out_queue
        self.sample_queue = sample_queue
        self.id = num

        self.proc = mp.Process(target=self._main, name=f"MemoryDecoder-{num}", daemon=daemon)

    def start(self):
        self.proc.start()

    def join(self):
        self.proc.join(timeout=1)

    def _parse_options(self, **kwargs):
        pass

    def _set_device(self):
        pass

    def _terminate(self):
        self.proc.terminate()
        self.proc.join()

    def __del__(self):
        self._terminate()

    def _main(self) -> None:
        """
        Decoder worker to be run alongside Learner. To save GPU memory, we leave it to the Learner to push tensors to
        GPU.
        """
        transition: Transition

        self.logger = get_logger_from_process(self.log_queue)
        self.logger.info(f"tid: {get_tid()} | Decoder process started")

        while True:
            batch = self.replay_out_queue.get()
            if self.replay_out_queue.empty():
                self.logger.debug(f'replay_out_queue EMPTY')
            if batch is None:  # end the process
                self.sample_queue.put(None)
                break

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


class Learner(BaseWorker):
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
                 replay_out_queues: List[torch.multiprocessing.Queue],
                 sample_queue: torch.multiprocessing.Queue,
                 model_params: DictProxy,
                 checkpoint_path: str,
                 log_queue: torch.multiprocessing.Queue,
                 learning_params: Dict[str, Union[float, int]],
                 n_decoders: int = 2,
                 run_profile: bool = False):
        """
        In two separate processes, decodes sampled data and runs training.

        :param n_decoders: Number of decoder processes to run
        :param optimizer: The selected type of optimizer
        :param model: The initialized model object to be copied into the learner
        :param replay_out_queues: _sample batches are pulled from this queue for decoding
        :param sample_queue: decoded batches are put on this queue for training
        :param model_params: a proxy to the model state dictionary
        :param checkpoint_path: Checkpoint save path
        :param log_queue: Queue object to be pushed to the log handler for the learner process
        :param learning_params: Parameters to control learning
        :param run_profile: If `True`, run optimizer with cProfile. 
        """
        self.replay_out_queues = replay_out_queues
        self.sample_queue = sample_queue
        self.log_queue = log_queue
        self._run_profile = run_profile

        self.checkpoint_path = checkpoint_path
        self._parse_options(**learning_params)

        self.n_decoder_processes = n_decoders
        self._set_device()

        self.policy = deepcopy(model).to(self.device)
        self.target = deepcopy(model).to(self.device)
        self.optimizer = optimizer(self.policy.parameters(), lr=self.learning_rate)

        self.model_params = model_params
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
        terminate and join the main process
        """
        self.main_proc.terminate()
        self.main_proc.join()

    def start(self) -> None:
        """
        Start all threads and processes
        """
        self.main_proc.start()
        del self.policy, self.target  # These have already been copied into the child process

    def _main(self) -> None:
        """
        The main worker process
        """
        # A started process has a `__weakref__` attribute that is not picklable. So, a started process cannot be passed
        # by context to another process. In this case, only one of the processes can be stored in `self`, and the other
        # process must be started before the `self` process.
        decoders = [Decoder(self.log_queue, q, self.sample_queue, i, daemon=True)
                    for i, q in enumerate(self.replay_out_queues)]
        [d.start() for d in decoders]

        self.policy_lock = threading.Lock()
        self.logger = get_logger_from_process(self.log_queue)
        self.logger.info(f"tid: {get_tid()} | Learner started on device {self.device}")

        param_update_thread = threading.Thread(target=self._copy_params, name='UpdateParams', daemon=True)
        param_update_thread.start()

        self._optimizer_loop()
        [d.join() for d in decoders]

    def _optimizer_loop(self) -> None:
        """
        Main training loop
        """
        interval_start_time = time.time()
        for self.epoch in count(1):
            batch = self._sample()
            if batch is None:
                break

            self._optimize_model(batch)
            if self.epoch % TARGET_UPDATE_INTERVAL == 0:
                self._update_target_net()
                self.logger.debug(f"{TARGET_UPDATE_INTERVAL} batches in {time.time() - interval_start_time} seconds")
                interval_start_time = time.time()
            if self.epoch % CHECKPOINT_INTERVAL == 0:
                self._save_checkpoint()

        self._save_checkpoint()

    def _copy_params(self) -> None:
        """
        Update the pipe every 2.5 seconds. Keep a lock to the pipe while it is being updated.
        """
        while True:
            with self.policy_lock:
                self.model_params.update(self._policy_state_dict_cpu())
            time.sleep(2.5)

    def _policy_state_dict_cpu(self):
        """
        Returns the policy state dict copied to cpu
        """
        cuda_state_dict = self.policy.state_dict()
        params_dict = OrderedDict()
        for k, v in cuda_state_dict.items():
            params_dict[k] = v.to('cpu', non_blocking=True)
        return params_dict

    def _optimize_model(self, batch: ProcessedBatch):
        """
        Run a batch through optimization
        """
        state_action_values = self._forward_policy(batch.actions, batch.states)
        next_state_values = self._forward_target(batch.non_final_mask, batch.non_final_next_states)
        expected_state_action_values = (next_state_values * self.gamma) + batch.rewards.float()

        loss = self._get_loss(state_action_values, expected_state_action_values)
        self._step_optimizer(loss)

    def _sample(self) -> ProcessedBatch:
        """
        Pull a batch from the sample queue and push to `device`
        :return: ProcessedBatch object on `device`
        """
        processed_batch = self.sample_queue.get()
        if self.sample_queue.empty():
            self.logger.debug(f'sample_queue EMPTY')
        if processed_batch is not None:
            processed_batch = processed_batch.to(self.device)
        return processed_batch

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
            {'policy_state': self.policy.state_dict()},
            os.path.join(self.checkpoint_path)
        )
