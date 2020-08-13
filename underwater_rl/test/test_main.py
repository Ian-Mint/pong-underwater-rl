import importlib
import math
import multiprocessing as mp
import os
import shutil
import sys
import time
import unittest
import unittest.mock as mock
from functools import reduce
from operator import mul

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

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


class TestEnv(unittest.TestCase):
    def setUp(self) -> None:
        set_main_args()

    def test_lstm_env_is_not_stacked_after_reset(self):
        main.ARCHITECTURE = 'lstm'
        env = main.dispatch_make_env(args)
        obs = env.reset()
        state = main.get_state(obs)
        self.assertEqual((1, 1, 84, 84), state.size())

    def test_default_env_has_4_frames_stacked_after_reset(self):
        main.ARCHITECTURE = None
        env = main.dispatch_make_env(args)
        obs = env.reset()
        state = main.get_state(obs)
        self.assertEqual((1, 4, 84, 84), state.size())

    def test_lstm_env_is_not_stacked_after_step(self):
        main.ARCHITECTURE = 'lstm'
        env = main.dispatch_make_env(args)
        _ = env.reset()
        obs, reward, done, info = env.step(0)
        state = main.get_state(obs)
        self.assertEqual((1, 1, 84, 84), state.size())

    def test_default_env_has_4_frames_stacked_after_step(self):
        main.ARCHITECTURE = None
        env = main.dispatch_make_env(args)
        _ = env.reset()
        obs, reward, done, info = env.step(0)
        state = main.get_state(obs)
        self.assertEqual((1, 4, 84, 84), state.size())


class TestMemoryEncoder(unittest.TestCase):
    def setUp(self) -> None:
        set_main_args()
        main.create_storage_dir(args.store_dir)
        main.logger = main.get_logger(main.args.store_dir)

        self.memory_queue, _, _, _, _, self.replay_out_queue, _ = main.get_communication_objects(args.actors)

        self.proc = mp.Process(target=main.memory_encoder, args=(self.memory_queue, self.replay_out_queue))

    def set_state(self, shape: tuple):
        x = np.linspace(0, 255, reduce(mul, shape)).reshape(shape).astype(np.uint8)
        self.state = torch.from_numpy(x)
        self.memory_queue.put(main.Transition(0, 0, self.state, 0, self.state, 0))

    def tearDown(self) -> None:
        self.proc.terminate()
        self.proc.join()

        shutil.rmtree(main.args.store_dir)

    def test_encoded_single_frame_image_is_png_format(self):
        self.set_state((1, 1, 100, 100))
        self.proc.start()
        _, _, f, _, _, _ = self.replay_out_queue.get()
        img = Image.open(f)
        self.assertEqual('PNG', img.format)

    def test_encoded_four_frame_image_is_png_format(self):
        self.set_state((1, 4, 100, 100))
        self.proc.start()
        _, _, f, _, _, _ = self.replay_out_queue.get()
        img = Image.open(f[0])
        self.assertEqual('PNG', img.format)

    def test_encoded_four_frame_image_is_decoded_to_a_torch_tensor(self):
        shape = (1, 4, 100, 100)
        self.set_state(shape)
        self.proc.start()
        _, _, f, _, _, _ = self.replay_out_queue.get()
        tensor = main.decode_stacked_frames(f)
        self.assertEqual(shape, tensor.size())


class TestModelInitialization(unittest.TestCase):
    def setUp(self) -> None:
        set_main_args()

    def tearDown(self) -> None:
        # noinspection PyTypeChecker
        importlib.reload(main)

    def assert_correct_initialization(self, model_class):
        policy = main.initialize_model(args.architecture)
        self.assertEqual(type(policy), model_class)

    def test_dqn_initialized_correctly(self):
        main.ARCHITECTURE = 'dqn'
        self.assert_correct_initialization(models.DQN)

    def test_lstm_initialized_correctly(self):
        main.ARCHITECTURE = 'lstm'
        self.assert_correct_initialization(models.DRQN)

    def test_distributional_initialized_correctly(self):
        main.ARCHITECTURE = 'distributional_dqn'
        self.assert_correct_initialization(models.DistributionalDQN)


class TestActor(unittest.TestCase):
    def setUp(self) -> None:
        set_main_args()
        set_main_constants()
        main.ARCHITECTURE = 'dqn'

        model = main.initialize_model(args.architecture)
        memory_queue, params_in, _, param_update_request, _, _, _ = main.get_communication_objects(args.actors)
        self.actor = main.Actor(model=model, n_episodes=10, render_mode=False, memory_queue=memory_queue,
                                replay_in_queue=replay_out_queue, pipe=None, global_args=, log_queue=, actor_params=)

        obs = self.actor.env.reset()
        self.state = main.get_state(obs)

    def tearDown(self) -> None:
        # noinspection PyTypeChecker
        importlib.reload(main)

    def assert_valid_action(self, action):
        self.assertIn(action, {0, 1, 2})

    def test_epsilon_is_1_at_start_of_training_episode_decay(self):
        main.STEPSDECAY = False
        eps = self.actor._epsilon
        self.assertEqual(1, eps)

    def test_epsilon_is_minimum_at_infinite_episodes_and_steps_episode_decay(self):
        main.STEPSDECAY = False
        self.actor.epoch = math.inf
        eps = self.actor._epsilon
        self.assertEqual(main.EPS_END, eps)

    def test_epsilon_is_1_at_start_of_training_steps_decay(self):
        main.STEPSDECAY = True
        eps = self.actor._epsilon
        self.assertEqual(1, eps)

    def test_epsilon_is_minimum_at_infinite_episodes_and_steps_steps_decay(self):
        main.STEPSDECAY = True
        self.actor.total_steps = math.inf
        eps = self.actor._epsilon
        self.assertEqual(main.EPS_END, eps)

    def test_random_action_chosen_at_start_of_training(self):
        with mock.patch.object(type(self.actor.policy), '__call__') as policy:
            _ = self.actor._select_action(self.state)
            self.assertFalse(policy.called)

    def test_valid_action_chosen_at_start_of_training(self):
        action = self.actor._select_action(self.state)
        self.assert_valid_action(action)

    def test_policy_net_action_chosen_at_end_of_training(self):
        type(self.actor)._epsilon = property(lambda *args: 0)  # mock _epsilon to 0
        with mock.patch.object(type(self.actor.policy), '__call__') as policy:
            _ = self.actor._select_action(self.state)
            self.assertTrue(policy.called)

    def test_policy_net_action_valid_at_end_of_training(self):
        type(self.actor)._epsilon = property(lambda *args: 0)  # mock _epsilon to 0
        action = self.actor._select_action(self.state)
        self.assert_valid_action(action)


class TestReplay(unittest.TestCase):
    def setUp(self) -> None:
        set_main_args()
        set_main_constants()

        _, _, _, replay_in_queue, replay_out_queue, _ = main.get_communication_objects(args.actors)
        self.replay = main.Replay(replay_in_queue, replay_out_queue, log_queue, )

    # todo: mock some queues and test multiprocessing


def first_greater_multiple(minimum, factor):
    return (minimum // factor + 1) * factor


class TestLearner(unittest.TestCase):
    def setUp(self) -> None:
        main.ARCHITECTURE = 'dqn'
        set_main_args()
        set_main_constants()
        self.store_dir = '__temp__'
        main.args.store_dir = self.store_dir
        main.logger = main.get_logger(self.store_dir)

        model = main.initialize_model(args.architecture)
        _, _, params_out, param_update_request, _, _, sample_queue = main.get_communication_objects(args.actors)
        self.learner = main.Learner(optimizer=optim.Adam, model=model, replay_out_queue=replay_out_queue,
                                    sample_queue=sample_queue, pipes=params_out, checkpoint_path=, log_queue=,
                                    learning_params=)

    def tearDown(self) -> None:
        del self.learner
        importlib.reload(main)
        if os.path.exists(self.store_dir):
            shutil.rmtree(self.store_dir)

    def test_learner_process_starts_and_blocks_with_empty_sample_queue(self):
        self.learner.start()
        time.sleep(1)
        if not self.learner.main_proc.is_alive():
            self.fail("Learner terminated unexpectedly")


@unittest.skip
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
