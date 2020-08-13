import random
from collections import namedtuple, deque

import numpy as np

__all__ = ['ReplayMemory', 'EpisodicMemory', 'PrioritizedReplay']

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

EPS = 1e-6


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def store(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class EpisodicMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.current_episode_memory = None
        self.memory = deque(maxlen=capacity)
        self.current_episode = None

    def store(self, episode: int, *args) -> None:
        """
        Stores completed episodes for sampling using `_sample`

        :param episode: current episode number
        :param args: ('state', 'action', 'next_state', 'reward')
        """
        if episode != self.current_episode:
            self.current_episode = episode
            if self.current_episode_memory is not None:
                self.memory.append(self.current_episode_memory)
            self.current_episode_memory = [Transition(*args)]
        else:
            self.current_episode_memory.append(Transition(*args))

    def sample(self, sequence_length, batch_size=1):
        assert batch_size == 1, "only a batch size of 1 is currently supported for LSTM"
        episodes_replay = random.sample(self.memory, batch_size)
        batch = []
        for e in episodes_replay:
            start = min(random.randint(0, len(e) - 1), len(e) - sequence_length)
            b = e[start:start + sequence_length]
            assert len(b) == sequence_length
            batch.append(b)
        return batch.pop()

    def __len__(self):
        return len(self.memory)


class PrioritizedReplay(object):
    def __init__(self,
                 capacity,
                 rank_based=False,
                 alpha=0.6,
                 beta=0.1,
                 beta_rate=1.000001):
        self.capacity = capacity
        self.memory = np.empty(shape=(self.capacity, 2), dtype=np.ndarray)
        self.n_entries = 0
        self.position = 0
        self.td_error_index = 0  # col 0 of memory
        self.sample_index = 1  # col 1 of memory
        self.rank_based = rank_based  # if not rank_based, then proportional
        self.alpha = alpha  # how much prioritization to use: 0 is uniform (no priority), 1 is full priority
        self.beta = beta  # bias correction 0 is no correction 1 is full correction
        self.beta_rate = beta_rate

    def update(self, idxs, td_errors):
        self.memory[idxs, self.td_error_index] = np.abs(td_errors)
        if self.rank_based:
            sorted_arg = self.memory[:self.n_entries, self.td_error_index].argsort()[::-1]
            self.memory[:self.n_entries] = self.memory[sorted_arg]

    def store(self, *args):
        priority = 1.0
        if self.n_entries > 0:
            priority = self.memory[:self.n_entries, self.td_error_index].max()
        self.memory[self.position, self.td_error_index] = priority
        self.memory[self.position, self.sample_index] = Transition(*args)
        self.n_entries = min(self.n_entries + 1, self.capacity)
        self.position = (self.position + 1) % self.capacity

    def _update_beta(self):
        """beta goes from beta0=0.1 to 1.0"""
        self.beta = min(1.0, self.beta * self.beta_rate)
        return self.beta

    def sample(self, batch_size):
        self._update_beta()
        entries = self.memory[:self.n_entries]

        if self.rank_based:
            priorities = 1 / (np.arange(self.n_entries) + 1)
        else:  # proportional
            priorities = entries[:, self.td_error_index] + EPS
        scaled_priorities = priorities ** self.alpha
        probs = np.array(scaled_priorities / np.sum(scaled_priorities), dtype=np.float64)

        weights = (self.n_entries * probs) ** -self.beta
        normalized_weights = weights / weights.max()
        idxs = np.random.choice(self.n_entries, batch_size, replace=False, p=probs)
        samples = np.array([entries[idx] for idx in idxs])

        samples_stacks = list(samples[:, self.sample_index])
        idxs_stack = np.vstack(idxs)
        weights_stack = np.vstack(normalized_weights[idxs])
        return idxs_stack, weights_stack, samples_stacks

    def __len__(self):
        return self.n_entries

    def __repr__(self):
        return str(self.memory[:self.n_entries])

    def __str__(self):
        return str(self.memory[:self.n_entries])
