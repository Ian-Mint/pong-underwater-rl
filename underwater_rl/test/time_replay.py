"""
144 samples pushed in 60s
"""
import multiprocessing as mp
import multiprocessing.queues as queues
import os
import pickle
import queue
import random
import sys
import threading
import time
from typing import List

try:
    import underwater_rl
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.pardir, os.path.pardir)))
from underwater_rl.main import get_communication_objects, get_logger, get_logger_from_process
from underwater_rl.replay import Replay
from underwater_rl.common import Transition
from underwater_rl.utils import get_tid

PATH_TO_DATA_STR = 'assets/memory.p'
replay_params = {
    'memory_size': 1000,
    'initial_memory': 1000,  # Wait for the memory buffer to fill before training
    'batch_size': 512
}


class InputQueueMock(queues.Queue):
    def __init__(self):
        with open(PATH_TO_DATA_STR, 'rb') as f:
            self._data = pickle.load(f)
        super().__init__(maxsize=0, ctx=_ctx)

    def __getstate__(self):
        """
        Put _data in state
        """
        mp.context.assert_spawning(self)
        return (self._ignore_epipe, self._maxsize, self._reader, self._writer,
                self._rlock, self._wlock, self._sem, self._opid, self._data)

    def __setstate__(self, state):
        """
        get _data from state
        """
        (self._ignore_epipe, self._maxsize, self._reader, self._writer,
         self._rlock, self._wlock, self._sem, self._opid, self._data) = state
        self._after_fork()

    def get(self, block=True, timeout=None):
        t = random.choice(self._data)
        if t is not None:
            return Transition(*t)
        else:
            return t

    def put(self, obj, block=True, timeout=None):
        raise NotImplementedError

    def terminate(self):
        with self._rlock:
            self._data = [None] * len(self._data)

    def empty(self):
        return False


class OutputQueueMock(queues.Queue):
    def __init__(self, maxsize=0):
        super().__init__(maxsize=maxsize, ctx=_ctx)
        self._qsize = 0

    def qsize(self):
        return self._qsize

    def get(self, block=True, timeout=None):
        with self._rlock:
            self._qsize -= 1
            if self.empty():
                raise queue.Empty

    def put(self, obj, block=True, timeout=None):
        with self._rlock:
            self._qsize += 1
            if self.full():
                raise queue.Full

    def full(self):
        return self._qsize >= self._maxsize

    def empty(self):
        return self._qsize <= 0


# noinspection PyProtectedMember
class TimeReplay:
    def __init__(self) -> None:
        self.in_q = InputQueueMock()
        self.out_q = OutputQueueMock(maxsize=0)
        self.log_q = mp.Queue()

        replay = Replay(self.in_q, self.out_q, self.log_q, replay_params)
        self.thread = threading.Thread(target=replay._main, name='Replay', daemon=True)

    def start(self):
        self.thread.start()

    def run(self):
        self.thread.run()

    def terminate(self, timeout=None):
        self.in_q.terminate()
        self.thread.join(timeout=timeout)


def test_replay_async(test_duration_s):
    tr = TimeReplay()
    tr.start()
    time.sleep(test_duration_s)
    tr.terminate(timeout=1.)


def push_worker(q: mp.Queue, sleep: float = 0):
    logger.info(f"tid: {get_tid()} | Push worker started")
    with open('assets/memory-np.p', 'rb') as f:
        samples = pickle.load(f)

    data = []
    for s in samples:
        data.append(Transition(*s))

    while True:
        s = random.choice(data)
        q.put(s)
        time.sleep(sleep)


def counter_worker(queue: List[mp.Queue], log_q: mp.Queue):
    logger = get_logger_from_process(log_q)

    timeout = 60.
    t_start = time.time()
    counter = 0
    logger.info(f"tid {get_tid()} | counter started")
    while time.time() - t_start < timeout:
        for i, q in enumerate(queue):
            counter += 1
            q.get()
            if q.empty():
                logger.debug(f"replay_out_queue-{i} EMPTY")
    logger.info(f"{counter} samples pushed in {timeout} seconds")


if __name__ == '__main__':
    mp.set_start_method('forkserver')
    _ctx = mp.get_context()

    comms = get_communication_objects(1, 1)
    logger, log_q = get_logger('tmp', mode='w')
    replay = Replay(replay_in_queue=comms.replay_in_q, replay_out_queues=comms.replay_out_q, log_queue=log_q,
                    params=replay_params)
    count_procs = [mp.Process(target=counter_worker, args=(comms.replay_out_q, log_q)) for _ in range(1)]
    push_thread = threading.Thread(target=push_worker, args=(comms.replay_in_q,), daemon=True)

    push_thread.start()
    replay.start()
    [p.start() for p in count_procs]

    [p.join() for p in count_procs]
    comms.replay_in_q.put(None)
    del replay
