import pickle
import random
import unittest

import numpy as np

from underwater_rl.common import Transition
from underwater_rl.replay import Memory

random.seed(0)
with open('assets/memory-np.p', 'rb') as f:
    data = pickle.load(f)
data = [Transition(*d) for d in data]


class TestMemory(unittest.TestCase):
    def setUp(self) -> None:
        self.max_length = 1000
        self.memory = Memory(length=self.max_length)

    def assert_transitions_equal(self, first: Transition, second: Transition):
        self.assertIsInstance(first, Transition)
        self.assertIsInstance(second, Transition)
        self.assertEqual(first.actor_id, second.actor_id)
        self.assertEqual(first.step_number, second.step_number)
        self.assertTrue(np.all(first.state == second.state))
        self.assertEqual(first.action, second.action)
        self.assertTrue(np.all(first.next_state == second.next_state))
        self.assertEqual(first.reward, second.reward)
        self.assertEqual(first.done, second.done)

    def test_sample_put_equals_sample_pulled(self):
        sample = random.choice(data)
        self.memory[0] = sample
        self.assert_transitions_equal(sample, self.memory[0])

    def test_sequence_put_equals_sequence_pulled(self):
        n_samples = 100
        samples = random.choices(data, k=n_samples)
        self.memory[:n_samples] = samples
        for s, m in zip(samples, self.memory[:n_samples]):
            self.assert_transitions_equal(s, m)


if __name__ == '__main__':
    unittest.main()
