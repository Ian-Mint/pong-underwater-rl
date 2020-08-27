import os
import random
import sys
import time
from multiprocessing.shared_memory import SharedMemory
from queue import Full, Empty
from typing import Dict, Union, List

import numpy as np
import torch.multiprocessing as mp

try:
    import underwater_rl
except ImportError:
    sys.path.append(os.path.abspath(os.path.pardir))
from underwater_rl.utils import get_logger_from_process, get_tid
from underwater_rl.common import Transition


class Memory:
    """
    A shared memory utility class
    """
    # n_bytes:
    #   2 4x84x84 uint8 arrays
    #   4 32-bit (4-byte) numbers
    #   1 bool (1-byte)
    int_size = 4
    array_dtype = 'uint8'
    array_bytes = 4 * 84 * 84
    array_shape = (4, 84, 84)
    stride = 2 * array_bytes + 4 * int_size + 1
    _offset = 0

    def __init__(self, length: int):
        self._length = length
        self._shared_memory = SharedMemory(create=True, size=self.stride * length)

        _n_locks = 20_000
        self._locks = [mp.Lock() for _ in range(_n_locks)]
        self._lock_length = length // _n_locks
        assert self._lock_length == length / _n_locks, "length must be divisible by _n_locks"

    def __del__(self):
        self._shared_memory.unlink()

    @property
    def _buf(self):
        return self._shared_memory.buf

    def __len__(self):
        return self._length

    def __getitem__(self, index: Union[slice, int]):
        if isinstance(index, int):
            return self._get_item(index)
        elif isinstance(index, slice):
            return self._get_slice(index)
        else:
            raise IndexError

    def _get_slice(self, slice_: slice):
        if slice_.stop > self._length:
            raise IndexError
        return [self._get_item(i % self._length) for i in range(slice_.start, slice_.stop, slice_.step)]

    # todo: use __get_slice__ and __set_slice__
    def _get_item(self, index):
        if index < 0 or index > self._length:
            raise IndexError(f"index {index} out of bounds")

        with self._locks[index // self._lock_length]:
            self._offset = index * self.stride

            actor_id = int.from_bytes(self._get(self.int_size), 'big')
            step_number = int.from_bytes(self._get(self.int_size), 'big')
            state = np.frombuffer(self._get(self.array_bytes), dtype='uint8').reshape(self.array_shape)
            action = int.from_bytes(self._get(self.int_size), 'big')
            next_state = np.frombuffer(self._get(self.array_bytes), dtype='uint8').reshape(self.array_shape)
            reward = int.from_bytes(self._get(self.int_size), 'big', signed=True)
            done = int.from_bytes(self._get(1), 'big')
            if done:
                next_state = None
            return Transition(actor_id, step_number, state, action, next_state, reward, done)

    def _get(self, n_bytes: int) -> memoryview:
        """
        Get item at `_offset` and move forward `n_bytes`

        :param n_bytes: Number of bytes to retrieve from memory
        :return: bytearray
        """
        item = self._buf[self._offset: self._offset + n_bytes]
        self._offset += n_bytes
        return item

    def __setitem__(self, index: Union[int, slice], transition: Union[List[Transition], Transition]):
        """
        Store `transition` in shared memory

        :param index: Index of the memory location to store
        :param transition: a `Transition`
        """
        if isinstance(index, int):
            assert isinstance(transition, Transition)
            self._set_item(index, transition)
        elif isinstance(index, slice):
            assert isinstance(transition, List)
            self._set_slice(index, transition)
        else:
            raise IndexError

    def _set_slice(self, slice_: slice, transitions: List[Transition]):
        step = slice_.step if slice_.step is not None else 1
        for i, t in zip(range(slice_.start, slice_.stop, step), transitions):
            self._set_item(i % self._length, t)

    def _set_item(self, index, transition):
        if index < 0 or index > self._length:
            raise IndexError(f"index {index} out of bounds")

        with self._locks[index // self._lock_length]:
            self._offset = index * self.stride

            # 'actor_id', 'step_number', 'state', 'action', 'next_state', 'reward', 'done'
            self._set(transition.actor_id.to_bytes(self.int_size, 'big'))
            self._set(transition.step_number.to_bytes(self.int_size, 'big'))
            self._set(transition.state.tobytes())
            self._set(transition.action.to_bytes(self.int_size, 'big'))
            if transition.next_state is not None:
                self._set(transition.next_state.tobytes())
            else:
                self._offset += self.array_bytes
            self._set(int(transition.reward).to_bytes(self.int_size, 'big', signed=True))
            self._set(transition.done.to_bytes(1, 'big'))

    def _set(self, bytearray_: Union[bytearray, bytes]):
        """
        update `_buf` and move `_offset`

        :param bytearray_: a bytearray
        """
        len_ = len(bytearray_)
        self._buf[self._offset: self._offset + len_] = bytearray_
        self._offset = self._offset + len_

    def __iter__(self):
        for i in range(self._length):
            yield self[i]


class Replay:
    r"""
                     +-------------+
    process          |    _main    |
                     +-------------+
                     /            \
                    /              \
    subprocess  _push_worker   _sample_worker
    """

    def __init__(self,
                 replay_in_queue: mp.Queue,
                 replay_out_queues: List[mp.Queue],
                 log_queue: mp.Queue,
                 params: Dict[str, Union[int, float]],
                 mode: str = 'default'):
        r"""
        Stores and samples the replay memory.

        :param replay_in_queue: The queue used to retrieve samples
        :param replay_out_queues: List of queues for pushing samples. One queue for each decoder
        :param params: parameters controlling the replay memory behavior (length, etc.)
        :param mode: {'default', 'episodic'}
        """
        # todo: implement episodic replay for use with LSTM
        if mode != 'default':
            raise NotImplementedError("Only default mode is currently implemented")

        self._parse_options(**params)

        assert self.initial_memory <= self.memory_maxlen, \
            "Initial replay memory set lower than total replay memory"

        self.replay_in_queue = replay_in_queue
        self.replay_out_queues = replay_out_queues
        self.log_queue = log_queue
        self.mode = mode

        self.buffer_in = []
        self.memory = Memory(self.memory_maxlen)
        self.memory_length = mp.Value('i', 0)
        self.sample_count = 0

        self.logger = None
        self.memory_full_event = mp.Event()
        self._main_proc = mp.Process(target=self._main, name="ReplaySample")

    def __del__(self):
        self._terminate()

    def _parse_options(self, memory_size, batch_size, initial_memory):
        self.memory_maxlen = memory_size
        self.initial_memory = initial_memory
        self.batch_size = batch_size

    def _set_device(self):
        raise NotImplementedError

    def _terminate(self):
        r"""
        Main thread
        """
        try:
            self.replay_in_queue.put(None, timeout=1)
            [q.put(None, timeout=1) for q in self.replay_out_queues]
        except (Full, Empty):
            pass
        self._main_proc.terminate()
        self._main_proc.join()

    def start(self):
        r"""
        Main thread
        """
        self._main_proc.start()

    def _main(self):
        r"""
        Launch push thread and run push worker in the main thread.
        """
        push_proc = mp.Process(target=self._push_worker, daemon=False, name="ReplayPush")
        sample_procs = [mp.Process(target=self._sample_worker, daemon=True, name=f"ReplaySampler-{i}", args=(q, ))
                        for i, q in enumerate(self.replay_out_queues)]
        push_proc.start()
        [p.start() for p in sample_procs]

        push_proc.join()
        [p.terminate() for p in sample_procs]

    def _push_worker(self) -> None:
        r"""
        Pushes from replay_in_queue to `memory`
        """
        self.logger = get_logger_from_process(self.log_queue)
        self.logger.info(f"tid: {get_tid()} | Replay memory push worker started")

        is_running = True
        while is_running:
            is_running = self._push()

    def _push(self):
        buffer_len = 1000
        sample = self.replay_in_queue.get()
        if sample is None:
            return False
        else:
            if self.replay_in_queue.empty():
                self.logger.debug(f'replay_in_queue EMPTY')

        self.buffer_in.append(sample)
        if len(self.buffer_in) >= buffer_len:
            index = self.sample_count % self.memory_maxlen
            self.memory[index: index + buffer_len] = self.buffer_in

            self.sample_count += buffer_len
            if not self.memory_full_event.is_set() and self.sample_count >= self.initial_memory:
                self.memory_full_event.set()
            self.buffer_in = []
        return True

    def _sample_worker(self, q: mp.Queue) -> None:
        r"""
        Generates samples from memory of length `batch_size` and pushes to `replay_out_queue`
        """
        self.logger = get_logger_from_process(self.log_queue)
        self.logger.info(f"tid: {get_tid()} | Replay memory sample worker started")
        self._wait_for_full_memory()

        while True:
            self._sample(q)

    def _sample(self, q: mp.Queue):
        batch = random.choices(self.memory, k=self.batch_size)
        q.put(batch)
        if q.full():
            self.logger.debug(f'replay_out_queue FULL')

    def _wait_for_full_memory(self) -> None:
        """
        Blocks until the memory length surpasses `initial_memory`
        """
        while not self.memory_full_event.is_set():
            time.sleep(1)
