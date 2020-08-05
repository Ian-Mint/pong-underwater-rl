#!/usr/bin/env python3
import argparse
import io
import logging
import math
import multiprocessing as mp
import os
import pickle
import random
import shutil
import sys
import threading
import time
from collections import namedtuple, deque
from copy import deepcopy
from itertools import count
from typing import Union, List

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from matplotlib import pyplot as plt
from torch.distributions import Categorical
from torchvision import transforms

sys.path.append(os.path.dirname(__file__))

try:
    from .memory import *
    from .models import *
    from .wrappers import *
    from .utils import convert_images_to_video, distr_projection, get_args_status_string, get_logger
except ImportError:
    from memory import *
    from models import *
    from wrappers import *
    from utils import convert_images_to_video, distr_projection, get_args_status_string, get_logger

# Constants
MEMORY_BATCH_SIZE = 100
N_ACTIONS = 3
ACTOR_UPDATE_INTERVAL = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
# warnings.filterwarnings("ignore", category=UserWarning)
Transition = namedtuple('Transition', ('actor_id', 'step_number', 'state', 'action', 'next_state', 'reward'))
ProcessedBatch = namedtuple(
    'ProcessedBatch', ('actions', 'rewards', 'states', 'non_final_mask', 'non_final_next_states', 'idxs', 'weights'))
HistoryElement = namedtuple('HistoryElement', ('n_steps', 'total_reward'))


class Learner:
    def __init__(self, optimizer, model, sample_queue: mp.Queue, event: mp.Event, params_out: mp.connection.Connection,
                 batch_size: int):
        if not isinstance(model, DQN):
            raise NotImplementedError("Only DQN models are implemented for now")

        # todo: currently, batch_size does not control the batch size. It is only used for calculating total samples
        self.sample_queue = sample_queue
        self.event = event
        self.params_out = params_out
        self.batch_size = batch_size

        self.policy = deepcopy(model)
        self.target = deepcopy(model)
        self.optimizer = optimizer(self.policy.parameters(), lr=LR)

        self.loss = None
        self.epoch = 0

        self.proc = mp.Process(target=self.main_worker, name="Learner")

    def __del__(self):
        self.save_checkpoint()
        self.terminate()

    def terminate(self):
        logger.debug("Learner terminating")
        self.proc.terminate()
        self.proc.join()

    def start(self):
        self.proc.start()
        logger.debug("Learner process started")

    def main_worker(self):
        for self.epoch in count(1):
            self.optimize_model()
            self.update_target_net()
            self.update_actors()
            if self.epoch % CHECKPOINT_INTERVAL:
                self.save_checkpoint()

    def update_actors(self):
        if self.event.is_set():
            self.params_out.send(self.policy.state_dict())
            self.event.clear()

    def optimize_model(self):
        action_batch, reward_batch, state_batch, non_final_mask, non_final_next_states, idxs, weights = self.sample()

        state_action_values = self.forward_policy(action_batch, state_batch)
        next_state_values = self.forward_target(non_final_mask, non_final_next_states)
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch.float()

        self.loss = get_loss(state_action_values, expected_state_action_values, idxs, weights)
        logger.debug(f"loss norm: {self.loss.norm()}")
        self.step_optimizer()

    def optimize_lstm(self):
        action_batch, reward_batch, state_batch, non_final_mask, non_final_next_states, idxs, weights = self.sample()

        self.policy.zero_hidden()
        self.target.zero_hidden()

        state_action_values = self.forward_policy(action_batch, state_batch)
        next_state_values = self.forward_target(non_final_mask, non_final_next_states)
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch.float()

        self.loss = get_loss(state_action_values, expected_state_action_values, idxs, weights)
        self.step_optimizer()

    def optimize_distributional(self):
        action_batch, reward_batch, state_batch, non_final_mask, non_final_next_states, idxs, weights = self.sample()

        next_states_v = torch.cat(
            [s if s is not None else state_batch[i] for i, s in enumerate(state_batch)]).to(DEVICE)
        dones = np.stack(tuple(map(lambda s: s is None, state_batch)))
        # next state distribution
        next_distr_v, next_qvals_v = self.target.both(next_states_v)
        next_actions = next_qvals_v.max(1)[1].data.cpu().numpy()
        next_distr = self.target.apply_softmax(next_distr_v).data.cpu().numpy()
        next_best_distr = next_distr[range(BATCH_SIZE), next_actions]
        dones = dones.astype(np.bool)
        # project our distribution using Bellman update
        proj_distr = distr_projection(next_best_distr, reward_batch.cpu().numpy(), dones,
                                      self.policy.Vmin, self.policy.Vmax, self.policy.atoms, GAMMA)
        # calculate net output
        distr_v = self.policy(state_batch)
        state_action_values = distr_v[range(BATCH_SIZE), action_batch.view(-1)]
        state_log_sm_v = F.log_softmax(state_action_values, dim=1)
        proj_distr_v = torch.tensor(proj_distr).to(DEVICE)
        entropy = (-state_log_sm_v * proj_distr_v).sum(dim=1)

        if PRIORITY:  # KL divergence based priority
            raise NotImplementedError("Prioritized replay not implemented")
        else:
            self.loss = entropy.mean()

        self.step_optimizer()

    def sample(self) -> ProcessedBatch:
        if PRIORITY:
            raise NotImplementedError("Prioritized replay not yet implemented")
        else:
            processed_batch = self.sample_queue.get()
            logger.debug(f'sample_queue length: {self.sample_queue.qsize()} after get')

        return processed_batch

    def forward_policy(self, action_batch, state_batch):
        return self.policy(state_batch).gather(1, action_batch)

    def forward_target(self, non_final_mask, non_final_next_states):
        next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
        if DOUBLE:
            argmax_a_q_sp = self.policy(non_final_next_states).max(1)[1]
            q_sp = self.target(non_final_next_states).detach()
            # noinspection PyTypeChecker
            next_state_values[non_final_mask] = q_sp[
                torch.arange(torch.sum(non_final_mask), device=DEVICE), argmax_a_q_sp]
        elif ARCHITECTURE == 'soft_dqn':
            raise NotImplementedError("soft DQN is not yet implemented")
        else:
            next_state_values[non_final_mask] = self.target(non_final_next_states).max(1)[0].detach()
        return next_state_values

    def step_optimizer(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        self.target.load_state_dict(self.policy.state_dict())

    def save_checkpoint(self):
        torch.save(
            {'policy_state': self.policy.state_dict(),
             'optimizer_state': self.optimizer.state_dict(),
             'samples_processed': self.epoch * self.batch_size,
             'epoch': self.epoch},
            os.path.join(args.store_dir, 'dqn'))


def separate_batches(actions, batch, rewards):
    state_batch = torch.cat(batch.state).to(DEVICE)
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)
    return action_batch, reward_batch, state_batch


def mask_non_final(batch):
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                  device=DEVICE, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(DEVICE)
    return non_final_mask, non_final_next_states


def process_transitions(transitions):
    batch = Transition(*zip(*transitions))
    actions = tuple((map(lambda a: torch.tensor([[a]], device=DEVICE), batch.action)))
    rewards = tuple((map(lambda r: torch.tensor([r], device=DEVICE), batch.reward)))
    return batch, actions, rewards


def get_state(obs):
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    return state.unsqueeze(0)


def get_loss(state_action_values, expected_state_action_values, idxs, weights):
    if PRIORITY:  # TD error based priority
        raise NotImplementedError("Prioritized replay not implemented")
    else:
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    return loss


class Actor:
    counter = 0

    def __init__(self, model: nn.Module, n_episodes: int, render_mode: Union[str, bool], memory_queue: mp.Queue,
                 load_params_event: mp.Event, params_in: mp.connection.Connection):
        """
        Main training loop

        :param load_params_event:
        :param params_in:
        :param n_episodes: Number of episodes over which to train or test
        :param render_mode: How and whether to visibly render states
        """
        self.params_in = params_in
        self.load_params_event = load_params_event
        self.memory_queue = memory_queue
        self.env = dispatch_make_env()
        self.policy = deepcopy(model)
        self.render_mode = render_mode
        self.n_episodes = n_episodes

        self.id = self.counter
        type(self).counter += 1

        self.episode = 0
        self.total_steps = 0
        self.history = []

        self.proc = mp.Process(target=self.main_worker, name=f"Actor-{self.id}")

    def __del__(self):
        logger.debug(f"Actor-{self.id} terminating")
        if self.proc.pid is not None:
            self.proc.terminate()
            self.proc.join()

    def join(self):
        self.proc.join()

    def start(self):
        self.proc.start()
        logger.debug(f"Actor-{self.id} started")

    def main_worker(self):
        for self.episode in range(1, self.n_episodes + 1):
            self.run_episode()
            logger.debug(f"Actor-{self.id}, episode {self.episode} complete")
            self.log_checkpoint(interval=LOG_INTERVAL)
        self.env.close()
        self.finish_rendering()
        logger.info(f"Actor-{self.id} done")

    def update_params(self):
        self.load_params_event.set()
        wait_event_not_set(self.load_params_event, timeout=None)
        logger.debug(f"params in:")
        params = self.params_in.recv()
        self.policy.load_state_dict(params)
        logger.debug(f"Actor-{self.id} params updated")

    def run_episode(self):
        obs = self.env.reset()
        state = get_state(obs)
        assert state.size() == (1, 4, 84, 84), logger.error(f"state is unexpected size: {state.size()}")
        total_reward = 0.0
        for steps in count():
            done, total_reward, state = self.run_step(state, total_reward)
            if self.total_steps % ACTOR_UPDATE_INTERVAL == 0:
                self.update_params()
            if done:
                break
        # noinspection PyUnboundLocalVariable
        self.history.append(HistoryElement(steps, total_reward))

    def run_step(self, state, total_reward):
        self.total_steps += 1

        action = self.select_action(state)
        self.dispatch_render()
        obs, reward, done, info = self.env.step(action)
        total_reward += reward
        if not done:
            next_state = get_state(obs)
        else:
            next_state = None
        if not args.test:
            self.memory_queue.put(Transition(self.id, self.total_steps, state, action, next_state, reward))
            logger.debug(f'memory_queue length: {self.memory_queue.qsize()} after put')
        return done, total_reward, next_state

    def select_action(self, state):
        if ARCHITECTURE == 'soft_dqn':
            return self.select_soft_action(state)
        else:
            return self.select_e_greedy_action(state)

    def select_e_greedy_action(self, state) -> int:
        if random.random() > self.epsilon:
            with torch.no_grad():
                if ARCHITECTURE == 'distribution_dqn':
                    return (self.policy.qvals(state.to(DEVICE))).argmax().item()
                else:
                    return self.policy(state.to(DEVICE)).max(1)[1].item()
        else:
            return random.randrange(N_ACTIONS)

    @property
    def epsilon(self):
        # todo: parameterize so that each actor can explore at different rates
        if STEPSDECAY:
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                            math.exp(-1. * self.total_steps / 1000000)
        else:
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                            math.exp(-1. * self.episode / EPS_DECAY)
        return eps_threshold

    def select_soft_action(self, state) -> int:
        with torch.no_grad():
            q = self.policy.forward(state.to(DEVICE))
            v = self.policy.getV(q).squeeze()
            dist = torch.exp((q - v) / self.policy.alpha)
            dist = dist / torch.sum(dist)
            c = Categorical(dist)
            a = c.sample()
        return a.item()

    def log_checkpoint(self, interval):
        if self.episode % interval:
            avg_reward = sum([h[0] for h in self.history[-interval:]]) / interval
            avg_steps = int(sum([h[1] for h in self.history[-interval:]]) / interval)
            logger.info(f'Actor: {self.id:<3}\t'
                        f'Total steps: {self.total_steps:<9}\t'
                        f'Episode: {self.episode:<5}\t'
                        f'Avg reward: {avg_reward:.2f}\t'
                        f'Avg steps: {avg_steps}')

    def dispatch_render(self):
        if self.render_mode:
            self.env.render(mode=self.render_mode, save_dir=args.store_dir)
            time.sleep(0.02)

    def finish_rendering(self):
        """
        If `render_mode` set to 'png', convert stored images to a video.
        """
        # todo: test this functionality, and also that of storing the png images
        save_dir = os.path.join(args.store_dir, f'actor{self.id}', 'video')
        if self.render_mode == 'png':
            convert_images_to_video(image_dir=save_dir, save_dir=os.path.dirname(save_dir))
            shutil.rmtree(args.save_dir)


def wait_event_not_set(event, timeout=None):
    """
    Blocks until the event is *not* set. Raises TimeoutError if the timeout is reached.

    :param event: multiprocessing.Event to wait for
    :param timeout: timeout (s)
    """
    if timeout is None:
        timeout = math.inf

    start_time = time.time()
    elapsed = 0
    while event.is_set() and elapsed < timeout:
        time.sleep(0.01)
        elapsed = time.time() - start_time

    if elapsed >= timeout:
        raise TimeoutError


def memory_encoder(memory_queue, replay_in_queue):
    """
    Encoder worker to be run alongside Actors
    :param replay_in_queue:
    :param memory_queue:
    """
    logger.debug("Memory encoder process started")
    while True:
        actor_id, step_number, state, action, next_state, reward = memory_queue.get()
        logger.debug(f'memory_queue length: {memory_queue.qsize()} after get')
        assert isinstance(state, torch.Tensor), logger.error(f"state must be a Tensor, not {type(state)}")
        assert isinstance(next_state, (torch.Tensor, type(None))), \
            logger.error(f"next_state must be a Tensor or None, not{type(next_state)}")
        assert isinstance(action, int), logger.error(f"action must be an integer, not {type(action)}")
        assert isinstance(reward, (int, float)), logger.error(f"reward must be a float, not {type(reward)}")

        state = state.squeeze().numpy()
        if next_state is not None:
            next_state = next_state.squeeze().numpy()

        png_next_state = None
        if state.ndim == 2:
            png_state = io.BytesIO()
            png_next_state = io.BytesIO()
            Image.fromarray(state).save(png_state, format='png')
            if next_state is not None:
                Image.fromarray(next_state).save(png_next_state, format='png')
        else:
            png_state = encode_stacked_frames(state)
            if next_state is not None:
                png_next_state = encode_stacked_frames(next_state)

        replay_in_queue.put(Transition(actor_id, step_number, png_state, action, png_next_state, reward))
        logger.debug(f'replay_in_queue length: {replay_in_queue.qsize()} after put')


def encode_stacked_frames(state) -> List[io.BytesIO]:
    result = []
    for frame in state:
        f = io.BytesIO()
        Image.fromarray(frame).save(f, format='png')
        result.append(f)
    return result


def decode_stacked_frames(png_state: List[io.BytesIO]) -> torch.Tensor:
    transform = transforms.ToTensor()
    result = []
    for f in png_state:
        frame = transform(Image.open(f))
        result.append(frame.squeeze())
    return torch.stack(result).unsqueeze(0).to(DEVICE)


def memory_decoder(sample_queue, replay_out_queue):
    """
    Decoder worker to be run alongside Learner
    """
    logger.debug("Memory decoder process started")

    transform = transforms.ToTensor()
    while True:
        batch = replay_out_queue.get()
        logger.debug(f'replay_out_queue length: {replay_out_queue.qsize()} after get')

        next_state = None
        decoded_batch = []
        for transition in batch:
            actor_id, step_number, png_state, action, png_next_state, reward = transition
            if isinstance(png_state, list):
                state = decode_stacked_frames(png_state)
                if png_next_state is not None:
                    next_state = decode_stacked_frames(png_next_state)
            else:
                state = transform(Image.open(png_state)).to(DEVICE)
                if png_next_state is not None:
                    next_state = transform(Image.open(png_next_state)).to(DEVICE)
            decoded_batch.append(Transition(actor_id, step_number, state, action, next_state, reward))

        batch, actions, rewards = process_transitions(decoded_batch)
        non_final_mask, non_final_next_states = mask_non_final(batch)
        action_batch, reward_batch, state_batch = separate_batches(actions, batch, rewards)
        processed_batch = ProcessedBatch(action_batch, reward_batch, state_batch,
                                         non_final_mask, non_final_next_states,
                                         idxs=None, weights=None)

        sample_queue.put(processed_batch)
        logger.debug(f'sample_queue length: {sample_queue.qsize()} after put')


class Replay:
    """
                     +-------------+
    process          | main_worker |
                     +-------------+
                     /            \
                    /              \
    threads     push_worker   sample_worker
    """

    def __init__(self, replay_in_queue, replay_out_queue, mode='default'):
        """

        :param replay_in_queue:
        :param replay_out_queue:
        :param mode: {'default', 'episodic'}
        """
        if mode != 'default':
            raise NotImplementedError("Only default mode is currently implemented")

        self.replay_in_queue = replay_in_queue
        self.replay_out_queue = replay_out_queue
        self.mode = mode
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.push_thread = threading.Thread(target=self.push_worker, daemon=True, name="Push")
        # self.sample_thread = threading.Thread(target=self.sample_worker, daemon=True, name="Sample")
        self.proc = mp.Process(target=self.main_worker, name="Replay")

    def __del__(self):
        self.terminate()

    def terminate(self):
        logger.debug("Replay terminating")
        self.proc.terminate()
        self.proc.join()

    def start(self):
        self.proc.start()
        logger.debug("Replay process started")

    def main_worker(self):
        """
        Launch push thread and run push worker in the main thread.
        """
        self.push_thread.start()
        self.sample_worker()

    def push_worker(self):
        logger.debug("Replay memory push worker started")
        while True:
            self.memory.append(self.replay_in_queue.get())
            logger.debug(f'replay_in_queue length: {self.replay_in_queue.qsize()} after get')

    def sample_worker(self):
        logger.debug("Replay memory sample worker started")
        while True:
            if len(self.memory) >= INITIAL_MEMORY:
                batch = random.sample(self.memory, BATCH_SIZE)
                self.replay_out_queue.put(batch)  # blocks if the queue is full
                logger.debug(f'replay_out_queue length: {self.replay_out_queue.qsize()} after put')


def display_state(state: torch.Tensor):
    """
    Displays the passed state using matplotlib

    :param state: torch.Tensor
    :return:
    """
    np_state = state.numpy().squeeze()
    fig, axs = plt.subplots(1, len(np_state), figsize=(20, 5))
    for img, ax in zip(np_state, axs):
        ax.imshow(img, cmap='gray')
    fig.show()


def dispatch_make_env():
    env = gym.make(
        "gym_dynamic_pong:dynamic-pong-v1",
        max_score=20,
        width=args.width,
        height=args.height,
        default_speed=args.ball,
        snell_speed=args.snell,
        snell_width=args.snell_width,
        snell_change=args.snell_change,
        snell_visible=args.snell_visible,
        refract=not args.no_refraction,
        uniform_speed=args.uniform_speed,
        our_paddle_speed=args.paddle_speed,
        their_paddle_speed=args.paddle_speed,
        our_paddle_height=args.paddle_length,
        their_paddle_height=args.paddle_length,
        our_paddle_angle=math.radians(args.paddle_angle),
        their_paddle_angle=math.radians(args.paddle_angle),
        their_update_probability=args.update_prob,
        ball_size=args.ball_size,
        ball_has_volume=args.ball_volume,
        state_type=args.state,
    )

    if ARCHITECTURE == 'lstm':
        env = make_env(env, stack_frames=False, episodic_life=True, clip_rewards=True, max_and_skip=False)
    else:
        env = make_env(env, stack_frames=True, episodic_life=True, clip_rewards=True, max_and_skip=True)
    return env


def initialize_model() -> nn.Module:
    model_lookup = {'dqn': DQN,
                    'soft_dqn': softDQN,
                    'dueling_dqn': DuelingDQN,
                    'lstm': DRQN,
                    'distributional_dqn': DistributionalDQN}
    return model_lookup[ARCHITECTURE](n_actions=N_ACTIONS).to(DEVICE)


def load_checkpoint():
    logger.info("Loading the trained model")
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)

    policy_net.load_state_dict(checkpoint['Net'])
    optimizer.load_state_dict(checkpoint['Optimizer'])
    target_net.load_state_dict(policy_net.state_dict())

    steps_done = checkpoint['Steps_Done']
    epoch = checkpoint['Epoch']

    history = pickle.load(open(args.history, 'rb'))

    return steps_done, epoch, history


def initialize_history():
    global steps_done, epoch
    if args.resume:
        steps_done, epoch, history = load_checkpoint()
    else:
        history = []
    return history


def get_parser():
    parser = argparse.ArgumentParser(description='Dynamic Pong RL')

    '''environment args'''
    env_args = parser.add_argument_group('Environment', "Environment controls")
    env_args.add_argument('--width', default=160, type=int,
                          help='canvas width (default: 160)')
    env_args.add_argument('--height', default=160, type=int,
                          help='canvas height (default: 160)')
    env_args.add_argument('--ball', default=1.0, type=float,
                          help='ball speed (default: 1.0)')
    env_args.add_argument('--ball-size', dest='ball_size', default=2.0, type=float,
                          help='ball size (default: 2.0)')
    env_args.add_argument('--ball-volume', dest='ball_volume', action='store_true', default=False,
                          help='If set, the ball interacts as if it has volume')
    env_args.add_argument('--snell', default=1.0, type=float,
                          help='snell speed (default: 1.0)')
    env_args.add_argument('--no-refraction', dest='no_refraction', default=False, action='store_true',
                          help='set to disable refraction')
    env_args.add_argument('--uniform-speed', dest='uniform_speed', default=False, action='store_true',
                          help='set to disable a different ball speed in the Snell layer')
    env_args.add_argument('--snell-width', dest='snell_width', default=80.0, type=float,
                          help='snell speed (default: 80.0)')
    env_args.add_argument('--snell-change', dest='snell_change', default=0, type=float,
                          help='Standard deviation of the speed change per step (default: 0)')
    env_args.add_argument('--snell-visible', dest='snell_visible', default='none', type=str,
                          choices=['human', 'machine', 'none'],
                          help="Determine whether snell is visible to when rendering ('render') or to the agent and "
                               "when rendering ('machine')")
    env_args.add_argument('--paddle-speed', default=1.0, type=float,
                          help='paddle speed (default: 1.0)')
    env_args.add_argument('--paddle-angle', default=70, type=float,
                          help='Maximum angle the ball can leave the paddle (default: 70deg)')
    env_args.add_argument('--paddle-length', default=20, type=float,
                          help='paddle length (default: 20)')
    env_args.add_argument('--update-prob', dest='update_prob', default=0.4, type=float,
                          help='Probability that the opponent moves in the direction of the ball (default: 0.4)')
    env_args.add_argument('--state', default='binary', type=str, choices=['binary', 'color'],
                          help='state representation (default: binary)')

    '''RL args'''
    rl_args = parser.add_argument_group("Model", "Reinforcement learning model parameters")
    rl_args.add_argument('--learning-rate', default=1e-4, type=float,
                         help='learning rate (default: 1e-4)')
    rl_args.add_argument('--network', default='dqn',
                         choices=['dqn', 'soft_dqn', 'dueling_dqn', 'resnet18', 'resnet10', 'resnet12',
                                  'resnet14', 'lstm', 'distribution_dqn'],
                         help='choose a network architecture (default: dqn)')
    rl_args.add_argument('--double', default=False, action='store_true',
                         help='switch for double dqn (default: False)')
    rl_args.add_argument('--pretrain', default=False, action='store_true',
                         help='switch for pretrained network (default: False)')
    rl_args.add_argument('--test', default=False, action='store_true',
                         help='Run the model without training')
    rl_args.add_argument('--render', default=False, type=str, choices=['human', 'png'],
                         help="Rendering mode. Omit if no rendering is desired.")
    rl_args.add_argument('--epsdecay', default=1000, type=int,
                         help="epsilon decay (default: 1000)")
    rl_args.add_argument('--stepsdecay', default=False, action='store_true',
                         help="switch to use default step decay")
    rl_args.add_argument('--episodes', dest='episodes', default=4000, type=int,
                         help='Number of episodes to train for (default: 4000)')
    rl_args.add_argument('--replay', default=100_000, type=int,
                         help="change the replay mem size (default: 100,000)")
    rl_args.add_argument('--priority', default=False, action='store_true',
                         help='switch for prioritized replay (default: False)')
    rl_args.add_argument('--rankbased', default=False, action='store_true',
                         help='switch for rank-based prioritized replay (omit if proportional)')
    rl_args.add_argument('--batch-size', dest='batch_size', default=32, type=int,
                         help="network training batch size or sequence length for recurrent networks")

    '''resume args'''
    resume_args = parser.add_argument_group("Resume", "Store experiments / Resume training")
    resume_args.add_argument('--resume', dest='resume', action='store_true',
                             help='Resume training switch. (omit to start from scratch)')
    resume_args.add_argument('--checkpoint', default='model.torch',
                             help='Checkpoint to load if resuming (default: model.torch)')
    resume_args.add_argument('--history', default='history.p',
                             help='History to load if resuming (default: history.p)')
    resume_args.add_argument('--store-dir', dest='store_dir',
                             default=os.path.join('..', 'experiments', time.strftime("%Y-%m-%d %H.%M.%S")),
                             help='Path to directory to store experiment results (default: ./experiments/<timestamp>/')

    '''debug args'''
    debug_args = parser.add_argument_group("Debug")
    debug_args.add_argument('--debug', action='store_true', help='Debug mode')
    return parser


def create_storage_dir():
    if not os.path.exists(args.store_dir):
        os.makedirs(args.store_dir)


def main():
    global args, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TARGET_UPDATE, LR, INITIAL_MEMORY, MEMORY_SIZE, \
        DOUBLE, STEPSDECAY, PRIORITY, LOG_INTERVAL, CHECKPOINT_INTERVAL, ARCHITECTURE, PRETRAIN, logger

    parser = get_parser()
    args = parser.parse_args()
    create_storage_dir()

    # hyperparameters
    BATCH_SIZE = args.batch_size
    GAMMA = 0.99
    EPS_START = 1
    EPS_END = 0.02
    EPS_DECAY = args.epsdecay
    TARGET_UPDATE = 1000
    LR = args.learning_rate
    INITIAL_MEMORY = args.replay // 10
    MEMORY_SIZE = args.replay
    DOUBLE = args.double
    STEPSDECAY = args.stepsdecay
    PRIORITY = args.priority

    LOG_INTERVAL = 20  # number of episodes between logging
    CHECKPOINT_INTERVAL = 100  # number of epochs between storing a checkpoint

    ARCHITECTURE = args.network
    PRETRAIN = args.pretrain

    logger, logger_thread = get_logger(args.store_dir)
    logger_thread.start()
    logger.info(get_args_status_string(parser, args))
    logger.info(f'Device: {DEVICE}')

    # Get shared objects
    model = initialize_model()
    """
    Memory and sampling pipeline
    Actor -> memory_queue
                        |  
    Encoder             +-> replay_in_queue
                                          |
    Replay                                +-> memory
                                                   |
    Replay                                         +-> replay_out_queue
                                                                      |
    Decoder                                                           +-> sample_queue
                                                                                     |
    Learner                                                                          +-> sample
    """
    memory_queue, pipe_in, pipe_out, param_update_request, replay_in_queue, replay_out_queue, sample_queue = \
        get_communication_objects()

    # Create subprocesses
    # todo: try multiple actors
    actor = Actor(model=model, n_episodes=args.episodes, render_mode=args.render, memory_queue=memory_queue,
                  load_params_event=param_update_request, params_in=pipe_in)
    png_encoder_proc = mp.Process(target=memory_encoder, args=(memory_queue, replay_in_queue), name='PNG Encoder')
    png_decoder_proc = mp.Process(target=memory_decoder, args=(sample_queue, replay_out_queue), name='PNG Decoder')
    replay = Replay(replay_in_queue, replay_out_queue)
    learner = Learner(optimizer=optim.Adam, model=model, sample_queue=sample_queue, event=param_update_request,
                      params_out=pipe_out, batch_size=BATCH_SIZE)

    # Start subprocesses
    actor.start()
    png_encoder_proc.start()
    replay.start()
    png_decoder_proc.start()
    learner.start()

    try:
        # Join subprocess. actor is the only one that is not infinite.
        actor.join()
        logger.info("All actors finished")
    except KeyboardInterrupt:
        del actor
    finally:
        logger.info("terminating PNG Encoder")
        png_encoder_proc.terminate()
        logger.info("terminating PNG Decoder")
        png_decoder_proc.terminate()
        del replay
        del learner
        png_encoder_proc.join()
        png_decoder_proc.join()


def get_communication_objects():
    memory_queue = mp.Queue(maxsize=1000)
    replay_in_queue = mp.Queue(maxsize=1000)
    replay_out_queue = mp.Queue(maxsize=100)
    sample_queue = mp.Queue(maxsize=100)

    param_update_request = mp.Event()
    pipe_in, pipe_out = mp.Pipe()
    return memory_queue, pipe_in, pipe_out, param_update_request, replay_in_queue, replay_out_queue, sample_queue


global args, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TARGET_UPDATE, LR, INITIAL_MEMORY, MEMORY_SIZE, \
    DOUBLE, STEPSDECAY, PRIORITY, LOG_INTERVAL, CHECKPOINT_INTERVAL, ARCHITECTURE, PRETRAIN, logger

if __name__ == '__main__':
    main()
