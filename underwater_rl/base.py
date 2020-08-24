import abc
from collections import namedtuple

import torch
from torch import multiprocessing as mp


Transition = namedtuple('Transition', ('actor_id', 'step_number', 'state', 'action', 'next_state', 'reward', 'done'))
HistoryElement = namedtuple('HistoryElement', ('n_steps', 'total_reward'))


class BaseWorker(abc.ABC):
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


class ParamPipe:
    def __init__(self):
        """
        Creates and holds an event object and a multiprocessing connection pair.

        Utility class used for communicating between learner and actors.
        """
        self.event = torch.multiprocessing.Event()
        self.conn_in, self.conn_out = mp.Pipe(duplex=False)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors