import multiprocessing as mp
import os
import random
import threading
import time
from collections import deque
from typing import Dict, Union
import sys

sys.path.append(os.path.abspath(os.path.pardir))
from underwater_rl.utils import get_logger_from_process, get_tid


class Replay:
    r"""
                     +-------------+
    process          |    _main    |
                     +-------------+
                     /            \
                    /              \
    threads     _push_worker   _sample_worker
    """

    def __init__(self,
                 replay_in_queue: mp.Queue,
                 replay_out_queue: mp.Queue,
                 log_queue: mp.Queue,
                 params: Dict[str, Union[int, float]],
                 mode: str = 'default'):
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
        self.buffer_length = self.initial_memory // 100
        self.memory = deque(maxlen=params['memory_size'])

        self.logger = None
        self.push_thread = None  # can't pass Thread objects to a new process using spawn or forkserver
        self.proc = mp.Process(target=self._main, name="Replay")

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
        self.proc.terminate()
        self.proc.join()

    def start(self):
        r"""
        Main thread
        """
        self.proc.start()

    def _main(self):
        r"""
        Launch push thread and run push worker in the main thread.
        """
        self.logger = get_logger_from_process(self.log_queue)
        self.logger.info(f"tid: {get_tid()} | Replay process started")

        self.lock = threading.Lock()

        self.push_thread = threading.Thread(target=self._push_worker, daemon=True, name="ReplayPush")
        self.push_thread.start()
        self._sample_worker()  # run _sample worker in the main thread

    def _push_worker(self) -> None:
        r"""
        Pushes from replay_in_queue to `memory`
        """
        self.logger.info(f"tid: {get_tid()} | Replay memory push worker started")
        is_running = True
        while is_running:
            is_running = self._push()

    def _push(self):
        sample = self.replay_in_queue.get()
        if sample is None:
            return False
        if self.replay_in_queue.empty():
            self.logger.debug(f'replay_in_queue EMPTY')
        self.buffer_in.append(sample)
        if len(self.buffer_in) >= self.buffer_length:
            self.memory.extend(self.buffer_in)
            self.buffer_in = []
        return True

    def _sample_worker(self) -> None:
        r"""
        Generates samples from memory of length `batch_size` and pushes to `replay_out_queue`
        """
        self.logger.info(f"tid: {get_tid()} | Replay memory sample worker started")
        self._wait_for_full_memory()

        while True:
            self._sample()

    def _sample(self):
        batch = random.choices(self.memory, k=self.batch_size)
        self.replay_out_queue.put(batch)
        self.logger.debug(f'replay_out_queue {self.replay_out_queue.qsize()}')

    def _wait_for_full_memory(self) -> None:
        """
        Blocks until the memory length surpasses `initial_memory`
        """
        while True:
            with self.lock:
                memory_length = len(self.memory)
            if memory_length >= self.initial_memory:
                break
            else:
                time.sleep(1)
                self.logger.debug(f'memory length: {memory_length}')