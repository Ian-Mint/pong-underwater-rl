import importlib
import math
import os
import shutil
import sys
import unittest
import unittest.mock as mock
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim

try:
    import underwater_rl.main as main
    import underwater_rl.main as memory
    import underwater_rl.models as models
    import underwater_rl.utils as utils
except ImportError:
    sys.path.append(os.path.join('..'))
    import main as main
    import memory as memory
    import models as models
    import utils as utils


def set_main_args():
    main.args = lambda: None
    main.args.width = 160
    main.args.height = 160
    main.args.ball = 1.
    main.args.snell = 1.
    main.args.snell_width = 80
    main.args.snell_change = 0
    main.args.snell_visible = 'none'
    main.args.no_refraction = False
    main.args.uniform_speed = False
    main.args.paddle_speed = 1.
    main.args.paddle_length = 20
    main.args.paddle_angle = 70
    main.args.update_prob = 0.4
    main.args.ball_size = 2.0
    main.args.ball_volume = False
    main.args.state = 'binary'
    main.args.store_dir = '__temp__'
    main.args.test = False
    main.args.resume = False

    main.LR = 1E-4
    main.STEPSDECAY = False
    main.EPS_DECAY = 1000
    main.PRIORITY = False
    main.MEMORY_SIZE = 10000
    main.TARGET_UPDATE = 1000
    main.CHECKPOINT_INTERVAL = 100
    main.LOG_INTERVAL = 20
    main.BATCH_SIZE = 32
    main.DOUBLE = False
    main.GAMMA = 0.99

    main.INITIAL_MEMORY = main.MEMORY_SIZE // 10


def set_main_constants():
    main.EPS_START = 1
    main.EPS_END = 0.02


def randomize_weights(model):
    for name, p in model.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform_(p)
        elif 'bias' in name:
            nn.init.zeros_(p)


def make_env(architecture):
    main.ARCHITECTURE = architecture
    main.env = main.dispatch_make_env()


def initialize_models(architecture):
    main.ARCHITECTURE = architecture
    main.policy_net, main.target_net = main.get_models()
    main.optimizer = optim.Adam(main.policy_net.parameters(), lr=main.LR)
    main.memory = main.initialize_replay_memory()
    main.history = main.initialize_history()


class TestEnv(unittest.TestCase):
    def setUp(self) -> None:
        set_main_args()

    def test_lstm_env_is_not_stacked_after_reset(self):
        main.ARCHITECTURE = 'lstm'
        env = main.dispatch_make_env()
        obs = env.reset()
        state = main.get_state(obs)
        self.assertEqual((1, 1, 84, 84), state.size())

    def test_default_env_has_4_frames_stacked_after_reset(self):
        main.ARCHITECTURE = None
        env = main.dispatch_make_env()
        obs = env.reset()
        state = main.get_state(obs)
        self.assertEqual((1, 4, 84, 84), state.size())

    def test_lstm_env_is_not_stacked_after_step(self):
        main.ARCHITECTURE = 'lstm'
        env = main.dispatch_make_env()
        _ = env.reset()
        obs, reward, done, info = env.step(0)
        state = main.get_state(obs)
        self.assertEqual((1, 1, 84, 84), state.size())

    def test_default_env_has_4_frames_stacked_after_step(self):
        main.ARCHITECTURE = None
        env = main.dispatch_make_env()
        _ = env.reset()
        obs, reward, done, info = env.step(0)
        state = main.get_state(obs)
        self.assertEqual((1, 4, 84, 84), state.size())


class TestMemoryInitialization(unittest.TestCase):
    def setUp(self) -> None:
        set_main_args()

    def test_lstm_has_episodic_memory(self):
        main.ARCHITECTURE = 'lstm'
        self.assertIsInstance(main.initialize_replay_memory(), memory.EpisodicMemory)

    def test_dqn_has_default_memory(self):
        main.ARCHITECTURE = 'dqn_pong_model'
        self.assertIsInstance(main.initialize_replay_memory(), memory.ReplayMemory)


class TestModelInitialization(unittest.TestCase):
    def setUp(self) -> None:
        set_main_args()

    def assert_correct_initialization(self, model_class):
        main.policy_net, main.target_net = main.get_models()
        self.assertEqual(type(main.policy_net), model_class)
        self.assertEqual(type(main.target_net), model_class)
        self.assertTrue(*utils.models_are_equal(main.policy_net, main.target_net))

    def test_dqn_initialized_correctly(self):
        make_env('dqn_pong_model')
        self.assert_correct_initialization(models.DQN)

    def test_lstm_initialized_correctly(self):
        make_env('lstm')
        self.assert_correct_initialization(models.DRQN)

    def test_distributional_initialized_correctly(self):
        make_env('distribution_dqn')
        self.assert_correct_initialization(models.DistributionalDQN)


class TestSelectAction(unittest.TestCase):
    def setUp(self) -> None:
        set_main_args()
        set_main_constants()
        main.steps_done = 0
        main.epoch = 0

        make_env('dqn_pong_model')
        obs = main.env.reset()
        self.state = main.get_state(obs)

        main.policy_net, main.target_net = main.get_models()

    def tearDown(self) -> None:
        try:
            main.get_epsilon = self.get_epsilon
        except AttributeError:
            pass

    def assert_valid_action(self, action):
        action = action.item()
        self.assertIn(action, {0, 1, 2})

    def test_get_epsilon_is_1_at_start_of_training_episode_decay(self):
        main.STEPSDECAY = False
        eps = main.get_epsilon(0, 0)
        self.assertEqual(1, eps)

    def test_get_epsilon_is_minimum_at_infinite_episodes_and_steps_episode_decay(self):
        main.STEPSDECAY = False
        eps = main.get_epsilon(math.inf, math.inf)
        self.assertEqual(main.EPS_END, eps)

    def test_get_epsilon_is_1_at_start_of_training_steps_decay(self):
        main.STEPSDECAY = True
        eps = main.get_epsilon(0, 0)
        self.assertEqual(1, eps)

    def test_get_epsilon_is_minimum_at_infinite_episodes_and_steps_steps_decay(self):
        main.STEPSDECAY = True
        eps = main.get_epsilon(math.inf, math.inf)
        self.assertEqual(main.EPS_END, eps)

    def test_random_action_chosen_at_start_of_training(self):
        with mock.patch.object(type(main.policy_net), '__call__') as policy_net:
            _ = main.select_action(self.state)
            self.assertFalse(policy_net.called)

    def test_valid_action_chosen_at_start_of_training(self):
        action = main.select_action(self.state)
        self.assert_valid_action(action)

    def test_policy_net_action_chosen_at_end_of_training(self):
        self.get_epsilon = main.get_epsilon
        main.get_epsilon = lambda *args: 0  # mock epsilon to 0
        with mock.patch.object(type(main.policy_net), '__call__') as policy_net:
            _ = main.select_action(self.state)
            self.assertTrue(policy_net.called)

    def test_policy_net_action_valid_at_end_of_training(self):
        self.get_epsilon = main.get_epsilon
        main.get_epsilon = lambda *args: 0  # mock epsilon to 0
        action = main.select_action(self.state)
        self.assert_valid_action(action)


class TestTrainFunctions(unittest.TestCase):
    def setUp(self) -> None:
        set_main_args()
        set_main_constants()
        main.steps_done = 0
        main.epoch = 0

        self.initialize()

    def initialize(self, architecture='dqn_pong_model'):
        make_env(architecture)
        obs = main.env.reset()
        self.state = main.get_state(obs)
        initialize_models(architecture)

    def test_models_equal_after_update_target_net_if_steps_done_is_multiple_of_target_update(self):
        randomize_weights(main.policy_net)
        main.update_target_net()
        self.assertTrue(*utils.models_are_equal(main.policy_net, main.target_net))

    def test_models_unequal_after_update_target_net_if_steps_done_is_not_multiple_of_target_update(self):
        main.steps_done = 1
        main.TARGET_UPDATE = 2
        randomize_weights(main.policy_net)
        main.update_target_net()
        self.assertFalse(utils.models_are_equal(main.policy_net, main.target_net)[0], "models are identical")


class TestMemory(unittest.TestCase):
    def setUp(self) -> None:
        set_main_args()
        set_main_constants()
        main.steps_done = 0
        main.epoch = 0

        self.initialize()

    def initialize(self, architecture='dqn_pong_model'):
        make_env(architecture)
        obs = main.env.reset()
        self.state = main.get_state(obs)
        initialize_models(architecture)

    def store_memory(self):
        self.episode = 0
        self.transition = memory.Transition(self.state, torch.tensor([[0]]), torch.tensor([0.]),
                                            torch.zeros_like(self.state))
        main.memory_store(self.episode, *self.transition)

    def test_lstm_has_episodic_memory(self):
        main.ARCHITECTURE = 'lstm'
        self.assertIsInstance(main.initialize_replay_memory(), memory.EpisodicMemory)

    def test_dqn_has_default_memory(self):
        main.ARCHITECTURE = 'dqn_pong_model'
        self.assertIsInstance(main.initialize_replay_memory(), memory.ReplayMemory)

    def test_models_equal_after_update_target_net_if_steps_done_is_multiple_of_target_update(self):
        randomize_weights(main.policy_net)
        main.update_target_net()
        self.assertTrue(*utils.models_are_equal(main.policy_net, main.target_net))

    def test_models_unequal_after_update_target_net_if_steps_done_is_not_multiple_of_target_update(self):
        main.steps_done = 1
        main.TARGET_UPDATE = 2
        randomize_weights(main.policy_net)
        main.update_target_net()
        self.assertFalse(utils.models_are_equal(main.policy_net, main.target_net)[0], "models are identical")

    def test_default_memory_store_increments_position(self):
        initial_memory = main.memory.position
        self.store_memory()
        self.assertEqual(initial_memory + 1, main.memory.position)

    def test_default_memory_store_transition_is_at_top_of_stack(self):
        self.store_memory()
        self.assertEqual(main.memory.memory.pop(), self.transition)

    def test_episodic_memory_store_transition_is_at_top_of_stack(self):
        self.initialize('lstm')
        self.store_memory()
        self.assertIsInstance(main.memory, memory.EpisodicMemory)
        self.assertEqual(main.memory.current_episode_memory.pop(), self.transition)


def first_greater_multiple(minimum, factor):
    return (minimum // factor + 1) * factor


class TestTrain(unittest.TestCase):
    def setUp(self) -> None:
        set_main_args()
        set_main_constants()
        main.steps_done = 0
        main.epoch = 0

        self.initialize()

    def tearDown(self) -> None:
        if os.path.exists(main.args.store_dir):
            shutil.rmtree(main.args.store_dir)

    def initialize(self, architecture='dqn_pong_model'):
        make_env(architecture)
        obs = main.env.reset()
        self.state = main.get_state(obs)
        initialize_models(architecture)

    def test_policy_net_is_not_updated_until_steps_greater_than_initial_memory(self):
        initial_net = deepcopy(main.policy_net)
        for i in range(max(main.INITIAL_MEMORY, main.BATCH_SIZE)):
            main.train_step(1, self.state, 0)
            self.assertTrue(*utils.models_are_equal(initial_net, main.policy_net))
        main.train_step(1, self.state, 0)
        self.assertFalse(utils.models_are_equal(initial_net, main.policy_net)[0])

    def test_policy_net_equals_target_net_after_target_update_steps(self):
        main.TARGET_UPDATE = 5
        for i in range(max(first_greater_multiple(main.INITIAL_MEMORY, main.TARGET_UPDATE),
                           first_greater_multiple(main.BATCH_SIZE, main.TARGET_UPDATE))):
            main.train_step(1, self.state, 0)
            if i >= max(main.INITIAL_MEMORY, main.BATCH_SIZE) and (i + 1) % main.TARGET_UPDATE != 0:
                self.assertFalse(utils.models_are_equal(main.target_net, main.policy_net)[0])
            elif i >= max(main.INITIAL_MEMORY, main.BATCH_SIZE) and (i + 1) % main.TARGET_UPDATE == 0:
                self.assertTrue(*utils.models_are_equal(main.target_net, main.policy_net))


class TestSystem(unittest.TestCase):
    def setUp(self) -> None:
        importlib.reload(main)
        self.store_dir = '__temp__'

    def tearDown(self) -> None:
        shutil.rmtree(self.store_dir)

    def test_main_runs_for_10_episodes_with_default_settings(self):
        sys.argv = [os.path.abspath('../main.py'), '--episodes', '110', '--store-dir', self.store_dir]
        main.main()


if __name__ == '__main__':
    unittest.main()
