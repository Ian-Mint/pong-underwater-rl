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
        self.assertEqual(first.state.dtype, second.state.dtype)
        self.assertEqual(first.state.dtype, second.state.dtype)

        self.assertEqual(first.action, second.action)
        self.assertTrue(np.all(first.next_state == second.next_state))
        self.assertEqual(first.reward, second.reward)
        self.assertEqual(first.done, second.done)

    def assert_transitions_not_equal(self, first: Transition, second: Transition):
        self.assertIsInstance(first, Transition)
        self.assertIsInstance(second, Transition)

        self.assertNotEqual(first.step_number, second.step_number)
        self.assertFalse(np.all(first.state == second.state))

    def test_sample_put_equals_sample_pulled(self):
        sample = data[0]
        self.memory[0] = sample
        self.assert_transitions_equal(sample, self.memory[0])

    def test_sample_put_and_changed_not_equal_sample_pulled(self):
        sample = data[0]
        self.memory[0] = sample
        assert not np.all(sample.state == data[1].state)
        sample.state[:] = data[1].state[:]
        self.assertFalse(np.all(sample.state == self.memory[0].state))

    def test_sample_put_not_equal_changed_sample_pulled(self):
        # todo: pass this test case
        sample = data[0]
        self.memory[0] = sample

        assert not np.all(sample.state == data[1].state)
        pulled = self.memory[0]
        self.memory[0] = data[1]
        self.assert_transitions_equal(pulled, sample)

    def test_sequence_put_equals_sequence_pulled(self):
        n_samples = 100
        samples = data[300:300 + n_samples]
        self.memory[:n_samples] = samples
        for s, m in zip(samples, self.memory[:n_samples]):
            self.assert_transitions_equal(s, m)

    def test_sequence_put_equals_sequence_pulled_after_one_fill_cycle(self):
        samples = data[:self.max_length + 1]
        for n, s in enumerate(samples[:-1]):
            self.memory[n] = s

        self.memory[0] = samples[-1]

        compare = [samples[-1]] + samples[1:]
        for x in zip(compare, self.memory):
            self.assert_transitions_equal(*x)

    def test_sequence_put_equals_sequence_pulled_after_two_fill_cycles(self):
        n_samples = 2000
        samples = data[:n_samples]
        for n, s in enumerate(samples):
            self.memory[n % self.max_length] = s

        for x in zip(data[n_samples - self.max_length:n_samples], self.memory):
            self.assert_transitions_equal(*x)


if __name__ == '__main__':
    unittest.main()
