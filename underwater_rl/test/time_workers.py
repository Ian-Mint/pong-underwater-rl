import os
import pickle
import queue
import random
import threading
import time
import multiprocessing as mp
import multiprocessing.queues as queues
import sys

try:
    from underwater_rl.replay import Replay
    from underwater_rl.base import Transition
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(os.curdir)))
    from replay import Replay
    from base import Transition

PATH_TO_DATA_STR = 'assets/memory.p'
_ctx = mp.get_context()


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
        replay_params = {
            'memory_size': 1000,
            'initial_memory': 1000,  # Wait for the memory buffer to fill before training
            'batch_size': 512
        }

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


def test_replay_queue_fill_time():
    pass


if __name__ == '__main__':
    test_replay_async(10)
