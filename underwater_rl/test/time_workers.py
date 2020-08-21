import pickle
import random
import threading
import time
import multiprocessing as mp

try:
    from ..replay import Replay
    from underwater_rl.base import Transition
except ImportError:
    from underwater_rl.replay import Replay
    from underwater_rl.utils import Transition


class InputQueueMock:
    def __init__(self, path_to_data: str):
        with open(path_to_data, 'rb') as f:
            self._data = pickle.load(f)

    def get(self, block=True, timeout=None):
        t = random.choice(self._data)
        if t is not None:
            return Transition(*random.choice(self._data))
        else:
            return t

    def put(self, obj, block=True, timeout=None):
        raise NotImplementedError

    def terminate(self):
        self._data[:] = [None] * len(self._data)

    def empty(self):
        return False


class OutputQueueMock:
    def __init__(self, maxlen=None):
        self.count = 0
        self.max_count = maxlen

    def put(self, obj, block=True, timeout=None):
        pass

    def full(self):
        return False


# noinspection PyProtectedMember
class TimeReplay:
    def __init__(self) -> None:
        replay_params = {
            'memory_size': 1000,
            'initial_memory': 1000,  # Wait for the memory buffer to fill before training
            'batch_size': 512
        }

        self.in_q = InputQueueMock('assets/memory.p')
        self.out_q = OutputQueueMock(maxlen=1000)
        self.log_q = mp.Queue()

        replay = Replay(self.in_q, self.out_q, self.log_q, replay_params)
        self.thread = threading.Thread(target=replay._main, name='Replay', daemon=True)

    def start(self):
        self.thread.start()

    def terminate(self, timeout=None):
        self.in_q.terminate()
        self.thread.join(timeout=timeout)


if __name__ == '__main__':
    tr = TimeReplay()
    tr.start()
    time.sleep(10)
    tr.terminate(timeout=1.)
