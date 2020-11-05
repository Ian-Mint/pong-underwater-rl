#!/usr/bin/env python3
import argparse
import logging
import math
import os
import pickle
import random
import shutil
import sys
import time
import warnings
from collections import namedtuple
from itertools import count

import gym
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

sys.path.append(os.path.dirname(__file__))

try:
    from .memory import *
    from .models import *
    from .wrappers import *
    from .utils import convert_images_to_video, distr_projection, get_args_status_string
    from .video_prediction import train_pong
except ImportError:
    from memory import *
    from models import *
    from wrappers import *
    from utils import convert_images_to_video, distr_projection, get_args_status_string
    from video_prediction import train_pong

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
warnings.filterwarnings("ignore", category=UserWarning)
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


def select_action(state):
    if architecture == 'soft_dqn':
        return select_soft_action(state)
    elif architecture == 'noisy_dqn':
        return select_noisy_action(state)
    else:
        return select_e_greedy_action(state)


def select_e_greedy_action(state):
    sample = random.random()
    eps_threshold = get_epsilon(epoch, steps_done)
    if sample > eps_threshold:
        with torch.no_grad():
            if architecture == 'distribution_dqn':
                return (policy_net.qvals(state.to(device))).argmax()
            elif architecture == 'predict_dqn':
                with torch.no_grad():
                    predict_model = train_pong.load_model(args.store_dir)
                    predicted_state = predict_model(state.unsqueeze(2).float()/255, future_seq=4).squeeze(1)*255
                    full_state_batch = torch.cat((state.float().to(device), predicted_state), dim=1)
                return policy_net(full_state_batch).max(1)[1]
            else:
                return policy_net(state.to(device)).max(1)[1]
    else:
        return torch.tensor([[random.randrange(env.action_space.n)]], device=device, dtype=torch.long)


def get_epsilon(epoch, steps_done):
    if STEPSDECAY:
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * steps_done / 1000000)
    else:
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * epoch / EPS_DECAY)
    return eps_threshold


def select_noisy_action(state):
    with torch.no_grad():
        return policy_net(state.to(device)).max(1)[1]


def select_soft_action(state):
    with torch.no_grad():
        q = policy_net.forward(state.to(device))
        v = policy_net.getV(q).squeeze()
        dist = torch.exp((q - v) / policy_net.alpha)
        dist = dist / torch.sum(dist)
        c = Categorical(dist)
        a = c.sample()
    return torch.tensor([[a.item()]], device=device, dtype=torch.long)


def optimize_model():
    batch, actions, rewards, idxs, weights = memory_sample()
    non_final_mask, non_final_next_states = mask_non_final(batch)
    action_batch, reward_batch, state_batch = separate_batches(actions, batch, rewards)

    state_action_values = forward_policy(action_batch, state_batch)
    next_state_values = forward_target(non_final_mask, non_final_next_states)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.float()

    loss = get_loss(state_action_values, expected_state_action_values, idxs, weights)
    step_optimizer(loss)


def optimize_lstm():
    batch, actions, rewards, idxs, weights = memory_sample()
    non_final_mask, non_final_next_states = mask_non_final(batch)
    action_batch, reward_batch, state_batch = separate_batches(actions, batch, rewards)

    policy_net.zero_hidden()
    target_net.zero_hidden()

    state_action_values = forward_policy(action_batch, state_batch)
    next_state_values = forward_target(non_final_mask, non_final_next_states)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.float()

    loss = get_loss(state_action_values, expected_state_action_values, idxs, weights)
    step_optimizer(loss)


def optimize_distributional():
    batch, actions, rewards, idxs, weights = memory_sample()
    action_batch, reward_batch, state_batch = separate_batches(actions, batch, rewards)

    next_states_v = torch.cat(
        [s if s is not None else batch.state[i] for i, s in enumerate(batch.next_state)]).to(device)
    dones = np.stack(tuple(map(lambda s: s is None, batch.next_state)))
    # next state distribution
    next_distr_v, next_qvals_v = target_net.both(next_states_v)
    next_actions = next_qvals_v.max(1)[1].data.cpu().numpy()
    next_distr = target_net.apply_softmax(next_distr_v).data.cpu().numpy()
    next_best_distr = next_distr[range(BATCH_SIZE), next_actions]
    dones = dones.astype(np.bool)
    # project our distribution using Bellman update
    proj_distr = distr_projection(next_best_distr, reward_batch.cpu().numpy(), dones,
                                  policy_net.Vmin, policy_net.Vmax, policy_net.atoms, GAMMA)
    # calculate net output
    distr_v = policy_net(state_batch)
    state_action_values = distr_v[range(BATCH_SIZE), action_batch.view(-1)]
    state_log_sm_v = F.log_softmax(state_action_values, dim=1)
    proj_distr_v = torch.tensor(proj_distr).to(device)
    entropy = (-state_log_sm_v * proj_distr_v).sum(dim=1)

    if PRIORITY:  # KL divergence based priority
        memory.update(idxs, entropy.detach().cpu().numpy().reshape(-1, 1))
        loss = (weights * entropy).mean()
    else:
        loss = entropy.mean()

    step_optimizer(loss)


def optimize_predict():
    batch, actions, rewards, idxs, weights = memory_sample()
    non_final_mask, non_final_next_states = mask_non_final(batch)
    action_batch, reward_batch, state_batch = separate_batches(actions, batch, rewards)     # (batch_size, 4, 84, 84), uint8
    # predict
    predict_model = train_pong.load_model(args.store_dir)
    with torch.no_grad():
        predicted_state_batch = predict_model(state_batch.unsqueeze(2).float()/255, future_seq=4).squeeze()*255
        full_state_batch = torch.cat((state_batch.float(), predicted_state_batch), dim=1)
        predicted_next_batch = predict_model(non_final_next_states.unsqueeze(2).float()/255, future_seq=4).squeeze()*255
        full_next_batch = torch.cat((non_final_next_states.float(), predicted_next_batch), dim=1)

    state_action_values = forward_policy(action_batch, full_state_batch)
    next_state_values = forward_target(non_final_mask, full_next_batch)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.float()

    loss = get_loss(state_action_values, expected_state_action_values, idxs, weights)
    step_optimizer(loss)


def forward_policy(action_batch, state_batch):
    return policy_net(state_batch).gather(1, action_batch)


# noinspection PyTypeChecker
def forward_target(non_final_mask, non_final_next_states):
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    if DOUBLE:
        argmax_a_q_sp = policy_net(non_final_next_states).max(1)[1]
        q_sp = target_net(non_final_next_states).detach()
        next_state_values[non_final_mask] = q_sp[
            torch.arange(torch.sum(non_final_mask), device=device), argmax_a_q_sp]
    elif architecture == 'soft_dqn':
        raise NotImplementedError("soft DQN is not yet implemented")
        # next_state_action_values[non_final_mask] = target_net(non_final_next_states).detach()
        # next_state_values[non_final_mask] = target_net.getV(next_state_action_values[non_final_mask]).detach()
    else:
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    return next_state_values


def step_optimizer(loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def separate_batches(actions, batch, rewards):
    state_batch = torch.cat(batch.state).to(device) # torch.Size([32, 4, 84, 84])
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)
    return action_batch, reward_batch, state_batch


def mask_non_final(batch):
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                  device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)
    return non_final_mask, non_final_next_states


def process_transitions(transitions):
    batch = Transition(*zip(*transitions))
    actions = tuple((map(lambda a: torch.tensor([[a]], device=device), batch.action)))
    rewards = tuple((map(lambda r: torch.tensor([r], device=device), batch.reward)))
    return batch, actions, rewards


def get_state(obs):
    state = np.array(obs)
    if state.ndim == 3:
        state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    return state.unsqueeze(0).unsqueeze(0)


def get_loss(state_action_values, expected_state_action_values, idxs, weights):
    if PRIORITY:  # TD error based priority
        td_errors = state_action_values - expected_state_action_values.unsqueeze(1)
        td_errors = td_errors.detach().cpu().numpy()
        memory.update(idxs, td_errors)
        loss = F.smooth_l1_loss(weights * state_action_values, weights * expected_state_action_values.unsqueeze(1))
    else:
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    return loss


def main_training_loop(n_episodes, render_mode=False):
    """
    Main training loop

    :param n_episodes: Number of episodes over which to train or test
    :param render_mode: How and whether to visibly render states
    """
    global epoch
    save_dir = os.path.join(args.store_dir, 'video')

    if args.train_prediction:
        train_pong.initial(args.store_dir, logger=logger)
    for episode in range(1, n_episodes + 1):
        train_episode(episode, render_mode, save_dir)
        if args.train_prediction and args.pred_episode < episode <= args.pred_episode+1000:
            if episode%20 == 1:
                logger.info(f'Start training prediction on {device} in episode {episode}~{episode+19}')
            train_prediction()
    env.close()
    finish_rendering(render_mode, save_dir)


def finish_rendering(render_mode, save_dir):
    """
    If `render_mode` set to 'png', convert stored images to a video.
    """
    if render_mode == 'png':
        convert_images_to_video(image_dir=save_dir, save_dir=os.path.dirname(save_dir))
        shutil.rmtree(save_dir)


def train_episode(episode, render_mode, save_dir):
    global epoch

    obs = env.reset()
    state = get_state(obs)  # torch.Size([1, 4, 84, 84])
    total_reward = 0.0
    t = 0
    for t in count():
        done, total_reward, state = train_step(episode, state, total_reward, render_mode, save_dir)

        if done:
            break
    epoch += 1

    save_episode(episode, t, total_reward)
    log_episode(episode, t)


def save_episode(episode, t, total_reward):
    history.append((total_reward, t))
    if episode % CHECKPOINT_INTERVAL == 0:
        save_checkpoint(args.store_dir)


def log_episode(episode, t):
    if episode % LOG_INTERVAL == 0:
        log_checkpoint(epoch, history, t)


def train_step(episode, state, total_reward, render_mode=False, save_dir=None):
    global steps_done
    steps_done += 1

    action = select_action(state)

    if args.debug:
        display_state(state, 'summed')
    dispatch_render(env, render_mode, save_dir)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    if not done:
        next_state = get_state(obs)
    else:
        next_state = None
    if not args.test:
        reward = torch.tensor([reward], device=device)
        memory_store(episode, state, action, next_state, reward)

        dispatch_optimize(episode)
    return done, total_reward, next_state


def dispatch_optimize(episode):
    if architecture == 'lstm':
        if episode >= INITIAL_MEMORY and len(memory) >= 1:
            optimize_lstm()
    else:
        if steps_done > INITIAL_MEMORY and len(memory) >= BATCH_SIZE:
            if architecture == 'distribution_dqn':
                optimize_distributional()
            elif architecture == 'predict_dqn':
                optimize_predict()
            else:
                optimize_model()

    update_target_net()


def memory_store(episode, state, action, next_state, reward):
    if isinstance(memory, EpisodicMemory):
        memory.store(episode, state, action.to('cpu'), next_state, reward.to('cpu'))
    else:
        memory.store(state, action.to('cpu'), next_state, reward.to('cpu'))


def memory_sample():
    idxs, weights = None, None

    if PRIORITY:
        idxs, weights, transitions = memory.sample(BATCH_SIZE)
        weights = torch.from_numpy(weights).float().to(device)
    else:
        transitions = memory.sample(BATCH_SIZE)

    batch, actions, rewards = process_transitions(transitions)
    return batch, actions, rewards, idxs, weights


def update_target_net():
    if steps_done % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())


def log_checkpoint(epoch, history, steps):
    avg_reward = sum([h[0] for h in history[-LOG_INTERVAL:]]) / LOG_INTERVAL
    avg_steps = int(sum([h[1] for h in history[-LOG_INTERVAL:]]) / LOG_INTERVAL)
    logger.info(f'Total steps: {steps_done}\tEpisode: {epoch}/{steps}\tAvg reward: {avg_reward:.2f}\t'
                f'Avg steps: {avg_steps}')


def display_state(state: torch.Tensor, mode='sequence'):
    """
    Displays the passed state using matplotlib

    :param state: torch.Tensor
    :return:
    """
    global fig, axs
    assert mode in {'sequence', 'summed'}

    np_state = state.numpy().squeeze()
    if mode == 'sequence':
        if 'fig' not in globals().keys():
            fig, axs = plt.subplots(1, len(np_state), figsize=(20, 5))
            for img, ax in zip(np_state, axs):
                ax.imshow(img, cmap='gray')
            fig.show()
        else:
            for img, ax in zip(np_state, axs):
                ax.imshow(img, cmap='gray')
            fig.canvas.draw()
    elif mode == 'summed':
        if 'fig' not in globals().keys():
            fig = plt.figure()
            axs = fig.gca()
            axs.imshow(np_state.sum(0), cmap='gray')
            if args.render != 'png':
                fig.show()
        else:
            axs.imshow(np_state.sum(0), cmap='gray')
            if args.render != 'png':
                fig.canvas.draw()

    if args.render == 'png':
        path = os.path.join(args.store_dir, 'video2')
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(os.path.join(path, f'{env.frame_count}.png'))


def dispatch_render(env, mode, save_dir):
    if mode:
        env.render(mode=mode, save_dir=save_dir)

def train_prediction():
    train_loader = train_pong.train_dataloader(replay=memory)
    training_loss = train_pong.training(dataloader=train_loader, store_dir=args.store_dir, learning_rate=LR, logger=logger)

def test_prediction():
    test_loader = train_pong.test_dataloader(replay=memory)
    training_loss = train_pong.testing(dataloader=test_loader, store_dir=args.store_dir)

def get_logger(store_dir):
    log_path = os.path.join(store_dir, 'output.log')
    logger = logging.Logger('train_status', level=logging.DEBUG)

    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(logging.Formatter('%(levelname)s\t%(message)s'))

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s\t%(levelname)s\t%(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    return logger


def save_checkpoint(store_dir):
    torch.save(
        {'Net': policy_net.state_dict(), 'Optimizer': optimizer.state_dict(), 'Steps_Done': steps_done, 'Epoch': epoch},
        os.path.join(store_dir, 'dqn_pong_model'))

    with open(os.path.join(store_dir, 'history.p'), 'wb') as f:
        pickle.dump(history, f)


def initialize_replay_memory():
    if PRIORITY:
        if args.rankbased:
            return PrioritizedReplay(MEMORY_SIZE, rank_based=True)
        else:
            return PrioritizedReplay(MEMORY_SIZE)
    else:
        if architecture == 'lstm':
            return EpisodicMemory(MEMORY_SIZE)
        else:
            return ReplayMemory(MEMORY_SIZE)


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

    # TODO: consider removing some of the wrappers - may improve performance
    if architecture == 'lstm':
        env = make_env(env, stack_frames=False, episodic_life=True, clip_rewards=True, max_and_skip=False)
    elif 'basic' in args.state:
        env = StackAndSkipEnv(env, skip=4)
    else:
        env = make_env(env, stack_frames=True, episodic_life=True, clip_rewards=True, max_and_skip=True)
    return env


def get_models(architecture, n_actions):
    if 'basic' in args.state:
        policy_net = BasicNet(in_channels=1, n_actions=n_actions).to(device)
        target_net = BasicNet(in_channels=1, n_actions=n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
    elif architecture == 'dqn_pong_model':
        policy_net = DQN(n_actions=n_actions).to(device)
        target_net = DQN(n_actions=n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
    elif architecture == 'soft_dqn':
        policy_net = softDQN(n_actions=n_actions).to(device)
        target_net = softDQN(n_actions=n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
    elif architecture == 'dueling_dqn':
        policy_net = DuelingDQN(n_actions=n_actions).to(device)
        target_net = DuelingDQN(n_actions=n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
    elif architecture == 'attention_dqn':
        policy_net = AttentionDQN(n_actions=n_actions).to(device)
        target_net = AttentionDQN(n_actions=n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
    elif architecture == 'lstm':
        policy_net = DRQN(n_actions=n_actions).to(device)
        target_net = DRQN(n_actions=n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
    elif architecture == 'distribution_dqn':
        policy_net = DistributionalDQN(n_actions=n_actions).to(device)
        target_net = DistributionalDQN(n_actions=n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
    elif architecture == 'predict_dqn':
        policy_net = DQN(in_channels=8, n_actions=n_actions).to(device)
        target_net = DQN(in_channels=8, n_actions=n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
    elif architecture == 'noisy_dqn':
        policy_net = NoisyDQN(n_actions=n_actions).to(device)
        target_net = NoisyDQN(n_actions=n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
    else:
        if architecture == 'resnet18':
            policy_net = resnet18(pretrained=pretrain)
            target_net = resnet18(pretrained=pretrain)
        elif architecture == 'resnet10':
            policy_net = resnet10()
            target_net = resnet10()
        elif architecture == 'resnet12':
            policy_net = resnet12()
            target_net = resnet12()
        elif architecture == 'resnet14':
            policy_net = resnet14()
            target_net = resnet14()
        else:
            raise ValueError('''Need an available architecture:
                    dqn_pong_model,
                    soft_dqn,
                    dueling_dqn,
                    distribution_dqn,
                    predict_dqn,
                    noisy_dqn,
                    lstm,
                    Resnet:
                        resnet18,
                        resnet10,
                        resnet12,
                        resnet14''')

        num_ftrs = policy_net.fc.in_features
        policy_net.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        policy_net.fc = nn.Linear(num_ftrs, n_actions)
        policy_net = policy_net.to(device)
        num_ftrs = target_net.fc.in_features
        target_net.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        target_net.fc = nn.Linear(num_ftrs, n_actions)
        target_net = target_net.to(device)
        target_net.load_state_dict(policy_net.state_dict())

    return policy_net, target_net


def load_checkpoint():
    logger.info("Loading the trained model")
    checkpoint = torch.load(args.checkpoint, map_location=device)

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
    env_args.add_argument('--state', default='binary', type=str,
                          choices=['color', 'binary', 'basic-render', 'basic-no-render', 'basic-binary'],
                          help='state representation (default: binary)')

    '''RL args'''
    rl_args = parser.add_argument_group("Model", "Reinforcement learning model parameters")
    rl_args.add_argument('--learning-rate', default=1e-4, type=float,
                         help='learning rate (default: 1e-4)')
    rl_args.add_argument('--network', default='dqn_pong_model',
                         choices=['dqn_pong_model', 'soft_dqn', 'dueling_dqn', 'resnet18', 'resnet10', 'resnet12',
                                  'resnet14', 'noisy_dqn', 'predict_dqn', 'lstm', 'distribution_dqn', 'attention_dqn'],
                         help='choose a network architecture (default: dqn_pong_model)')
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
    rl_args.add_argument('--train-prediction', dest='train_prediction', default=False, action='store_true',
                         help='train prediction(default: False)')
    rl_args.add_argument('--pred-episode', dest='pred_episode', default=1000, type=int,
                         help="when to start training prediction model")

    '''resume args'''
    resume_args = parser.add_argument_group("Resume", "Store experiments / Resume training")
    resume_args.add_argument('--resume', dest='resume', action='store_true',
                             help='Resume training switch. (omit to start from scratch)')
    resume_args.add_argument('--checkpoint', default='dqn_pong_model',
                             help='Checkpoint to load if resuming (default: dqn_pong_model)')
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
    # noinspection PyGlobalUndefined
    global args, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TARGET_UPDATE, LR, INITIAL_MEMORY, MEMORY_SIZE,\
        DOUBLE, STEPSDECAY, PRIORITY, LOG_INTERVAL, CHECKPOINT_INTERVAL, architecture, pretrain, logger, env, \
        policy_net, target_net, steps_done, epoch, optimizer, history, memory

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

    # number episodes between logging and saving
    LOG_INTERVAL = 20
    CHECKPOINT_INTERVAL = 100

    architecture = args.network
    pretrain = args.pretrain

    logger = get_logger(args.store_dir)
    logger.info(get_args_status_string(parser, args))
    logger.info(f'Device: {device}')

    env = dispatch_make_env()
    policy_net, target_net = get_models(architecture, env.action_space.n)

    steps_done = 0
    epoch = 0
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    history = initialize_history()
    memory = initialize_replay_memory()

    try:
        main_training_loop(args.episodes, render_mode=args.render)
    except KeyboardInterrupt:
        save_checkpoint(args.store_dir)


global args, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TARGET_UPDATE, LR, INITIAL_MEMORY, MEMORY_SIZE,\
    DOUBLE, STEPSDECAY, PRIORITY, LOG_INTERVAL, CHECKPOINT_INTERVAL, architecture, pretrain, logger, env, \
    policy_net, target_net, steps_done, epoch, optimizer, history, memory

if __name__ == '__main__':
    main()
