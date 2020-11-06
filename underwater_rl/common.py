import abc
import time
from collections import namedtuple
from typing import NamedTuple, List, Iterable

import torch
from torch import multiprocessing as mp

Transition = namedtuple('Transition', ('actor_id', 'step_number', 'state', 'action', 'next_state', 'reward', 'done'))
HistoryElement = namedtuple('HistoryElement', ('n_steps', 'total_reward'))


class Comms(NamedTuple):
    memory_q: mp.Queue
    replay_in_q: mp.Queue
    replay_out_q: List[mp.Queue]
    sample_q: mp.Queue


class BaseWorker(abc.ABC):
    """
    Base class for multiprocessing worker classes
    """

    @abc.abstractmethod
    def start(self): ...

    @abc.abstractmethod
    def is_alive(self) -> bool: ...

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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors


def run_all(workers: Iterable, method: str, *args, **kwargs) -> None:
    r"""
    Executes `method` for all actors in `actors`. Equivalent to
    ```
    for a in actors:
        a.method(*args, **kwargs)
    ```

    :param workers: Iterable of actor objects
    :param method: Actor.method to execute
    :param args:
    :param kwargs:
    """
    for a in workers:
        a.__getattribute__(method)(*args, **kwargs)


def join_first(workers: Iterable[BaseWorker]):
    """
    Block until one worker is complete. When that happens, terminate all other workers.
    """
    while all((w.is_alive() for w in workers)):
        time.sleep(1)

    run_all(workers, '__del__')
