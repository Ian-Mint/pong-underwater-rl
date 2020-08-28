import os
import random
import sys
import time
from queue import Full, Empty
from typing import Dict, Union, List

import torch.multiprocessing as mp
from torch.multiprocessing import Manager

try:
    import underwater_rl
except ImportError:
    sys.path.append(os.path.abspath(os.path.pardir))
from underwater_rl.utils import get_logger_from_process, get_tid
from underwater_rl.common import Transition


class Memory:
    _offset = 0

    def __init__(self, length: int, dest=None):
        self._length = length
        init = [None] * length
        if dest is None:
            self._memory = init
        else:
            self._memory = dest
            if len(dest) == 0:
                self._memory.extend(init)
            else:
                assert len(dest) == length, "dest must be empty or equal to length"

    def __len__(self):
        return self._length

    def __getitem__(self, index: Union[slice, int]):
        return self._memory[index]

    def __get_slice__(self, slice_: slice):
        return [self[i % self._length] for i in range(slice_.start, slice_.stop, slice_.step)]

    def put(self, transition):
        assert isinstance(transition, Transition)
        self._memory[self._offset] = transition
        self._offset = (self._offset + 1) % self._length

    def extend(self, transitions):
        for t in transitions:
            self.put(t)

    def __iter__(self):
        for m in self._memory:
            yield m


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
        self.memory = None
        self.memory_length = mp.Value('i', 0)
        self.sample_count = 0

        self.logger = None
        self.memory_full_event = mp.Event()
        self._main_proc = mp.Process(target=self._main, name="ReplayMain")

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
        Launch push process and run push worker in the main process.
        """
        with mp.Manager() as m:
            self.memory = Memory(self.memory_maxlen, m.list())
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

        self.buffer_in.append(sample)
        if len(self.buffer_in) >= buffer_len:
            self.memory.extend(self.buffer_in)

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
