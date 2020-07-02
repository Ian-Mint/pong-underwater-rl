#!/usr/bin/env python3
import argparse
import logging
import math
import os
import pickle
import random
import shutil
import time
import warnings
from collections import namedtuple
from itertools import count

import gym
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path.append(os.path.dirname(__file__))

from memory import ReplayMemory, PrioritizedReplay
from models import *
from wrappers import *
from utils import convert_images_to_video

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
warnings.filterwarnings("ignore", category=UserWarning)
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


def select_action(state):
    global steps_done
    global epoch
    sample = random.random()
    if STEPSDECAY:
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * steps_done / 1000000)
    else:
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * epoch / EPS_DECAY)

    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state.to(device)).max(1)[1]
    else:
        # TODO: should this just go the the CPU?
        return torch.tensor([[random.randrange(env.action_space.n)]], device=device, dtype=torch.long)


def tic():
    return time.time()


def toc(tstart, nm=""):
    print('%s took: %s sec.\n' % (nm, (time.time() - tstart)))


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    if PRIORITY:
        idxs, weights, transitions = memory.sample(BATCH_SIZE)
        weights = torch.from_numpy(weights).float().to(device)
    else:
        transitions = memory.sample(BATCH_SIZE)
    """
    zip(*transitions) unzips the transitions into
    Transition(*) creates new named tuple
    batch.state - tuple of all the states (each state is a tensor)
    batch.next_state - tuple of all the next states (each state is a tensor)
    batch.reward - tuple of all the rewards (each reward is a float)
    batch.action - tuple of all the actions (each action is an int)    
    """
    batch = Transition(*zip(*transitions))

    actions = tuple((map(lambda a: torch.tensor([[a]], device=device), batch.action)))
    rewards = tuple((map(lambda r: torch.tensor([r], device=device), batch.reward)))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.uint8
    )

    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None]).to(device)

    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    if DOUBLE:
        argmax_a_q_sp = policy_net(non_final_next_states).max(1)[1]
        q_sp = target_net(non_final_next_states).detach()
        next_state_values[non_final_mask] = q_sp[torch.arange(torch.sum(non_final_mask), device=device), argmax_a_q_sp]
    else:
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.float()

    if PRIORITY:
        td_errors = state_action_values - expected_state_action_values.unsqueeze(1)
        td_errors = td_errors.detach().cpu().numpy()
        memory.update(idxs, td_errors)
        loss = F.smooth_l1_loss(weights * state_action_values, weights * expected_state_action_values.unsqueeze(1))
    else:
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def get_state(obs):
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    return state.unsqueeze(0)


def train(env, n_episodes, history, render=False):
    global epoch
    for episode in range(1, n_episodes + 1):
        obs = env.reset()
        state = get_state(obs)  # torch.Size([1, 4, 84, 84])
        total_reward = 0.0
        for t in count():
            if render:
                save_dir = os.path.join(args.store_dir, 'video')
                env.render(mode=render, save_dir=save_dir)

            if args.random_agent:
                action = env.select_action()
                action = torch.tensor([action], device='cpu')
            else:
                action = select_action(state)
            obs, reward, done, info = env.step(action)

            total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            reward = torch.tensor([reward], device=device)

            memory.store(state, action.to('cpu'), next_state, reward.to('cpu'))
            state = next_state

            if steps_done > INITIAL_MEMORY:
                optimize_model()
                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        epoch += 1
        history.append((total_reward, t))
        if episode % LOG_INTERVAL == 0:
            avg_reward = sum([h[0] for h in history[-LOG_INTERVAL:]]) / LOG_INTERVAL
            avg_steps = int(sum([h[1] for h in history[-LOG_INTERVAL:]]) / LOG_INTERVAL)
            logger.info(f'Total steps: {steps_done}\tEpisode: {epoch}/{t}\tAvg reward: {avg_reward:.2f}\t'
                        f'Avg steps: {avg_steps}')
        if episode % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(args.store_dir)

    env.close()

    if render == 'png':
        convert_images_to_video(image_dir=save_dir, save_dir=os.path.dirname(save_dir))
        shutil.rmtree(save_dir)

    return history


def test(env, n_episodes, policy, render=True):
    # todo: look into using the Monitor wrapper
    save_dir = os.path.join(args.store_dir, 'video')
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for t in count():
            action = policy(state.to(device)).max(1)[1].view(1, 1)

            if render:
                env.render(mode=render, save_dir=save_dir)
                time.sleep(0.02)

            obs, reward, done, info = env.step(action)

            total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            state = next_state

            if done:
                logger.info("Finished Episode {} with reward {}".format(episode, total_reward))
                break

    env.close()
    if render == 'png':
        convert_images_to_video(image_dir=save_dir, save_dir=os.path.dirname(save_dir))
        shutil.rmtree(save_dir)


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
    global steps_done
    global epoch
    torch.save(
        {'Net': policy_net.state_dict(), 'Optimizer': optimizer.state_dict(), 'Steps_Done': steps_done, 'Epoch': epoch},
        os.path.join(store_dir, 'dqn_pong_model'))
    pickle.dump(history, open(os.path.join(store_dir, 'history.p'), 'wb'))


# noinspection PyProtectedMember
def get_args_status_string(parser: argparse.ArgumentParser, args: argparse.Namespace) -> str:
    """
    Returns a formatted string of the passed arguments.
    
    :param parser: The `argparse.ArgumentParser` object to read from
    :param args: The values of the parsed arguments from `parser.parse_args()`
    :return: "--width 160 --height 160 --ball 3.0 ..."
    """
    args_info = parser._option_string_actions
    s = ''
    for k, v in args_info.items():
        if isinstance(v, argparse._StoreAction):
            s += f'{k} {args.__dict__[v.dest]} '
    return s


if __name__ == '__main__':
    # arguments
    parser = argparse.ArgumentParser(description='Dynamic Pong RL')

    '''environment args'''
    env_args = parser.add_argument_group('Environment', "Environment controls")
    env_args.add_argument('--width', default=160, type=int,
                          help='canvas width (default: 160)')
    env_args.add_argument('--height', default=160, type=int,
                          help='canvas height (default: 160)')
    env_args.add_argument('--ball', default=3.0, type=float,
                          help='ball speed (default: 3.0)')
    env_args.add_argument('--ball-size', dest='ball_size', default=2.0, type=float,
                          help='ball size (default: 2.0)')
    env_args.add_argument('--snell', default=3.0, type=float,
                          help='snell speed (default: 3.0)')
    env_args.add_argument('--no-refraction', dest='no_refraction', default=False, action='store_true',
                          help='set to disable refraction')
    env_args.add_argument('--uniform-speed', dest='uniform_speed', default=False, action='store_true',
                          help='set to disable a different ball speed in the Snell layer')
    env_args.add_argument('--snell-width', dest='snell_width', default=40.0, type=float,
                          help='snell speed (default: 40.0)')
    env_args.add_argument('--snell-change', dest='snell_change', default=0, type=float,
                          help='Standard deviation of the speed change per step (default: 0)')
    env_args.add_argument('--snell-visible', dest='snell_visible', default='none', type=str,
                          choices=['human', 'machine', 'none'],
                          help="Determine whether snell is visible to when rendering ('render') or to the agent and "
                               "when rendering ('machine')")
    env_args.add_argument('--paddle-speed', default=3.0, type=float,
                          help='paddle speed (default: 3.0)')
    env_args.add_argument('--paddle-angle', default=45, type=float,
                          help='Maximum angle the ball can leave the paddle (default: 45deg)')
    env_args.add_argument('--paddle-length', default=45, type=float,
                          help='paddle length (default: 45)')
    env_args.add_argument('--update-prob', dest='update_prob', default=0.2, type=float,
                          help='Probability that the opponent moves in the direction of the ball (default: 0.2)')
    env_args.add_argument('--state', default='binary', type=str, choices=['binary', 'color'],
                          help='state representation (default: binary)')
    env_args.add_argument('--random-agent', dest='random_agent', default=False, action='store_true',
                          help="Use a random agent with update-probability 0.3")

    '''RL args'''
    rl_args = parser.add_argument_group("Model", "Reinforcement learning model parameters")
    rl_args.add_argument('--learning-rate', default=1e-4, type=float,
                         help='learning rate (default: 1e-4)')
    rl_args.add_argument('--network', default='dqn_pong_model',
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
    rl_args.add_argument('--replay', default=10000, type=int,
                         help="change the replay mem size (default: 10000)")
    rl_args.add_argument('--priority', default=False, action='store_true',
                         help='switch for prioritized replay (default: False)')
    rl_args.add_argument('--rankbased', default=False, action='store_true',
                         help='switch for rank-based prioritized replay (omit if proportional)')

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

    args = parser.parse_args()

    # create storage directory
    if not os.path.exists(args.store_dir):
        os.makedirs(args.store_dir)

    # hyperparameters
    BATCH_SIZE = 32
    GAMMA = 0.99
    EPS_START = 1
    EPS_END = 0.02
    EPS_DECAY = args.epsdecay
    TARGET_UPDATE = 1000
    RENDER = args.render
    lr = args.learning_rate
    INITIAL_MEMORY = args.replay
    MEMORY_SIZE = 10 * INITIAL_MEMORY
    DOUBLE = args.double
    STEPSDECAY = args.stepsdecay
    PRIORITY = args.priority

    # number episodes between logging and saving
    LOG_INTERVAL = 20
    CHECKPOINT_INTERVAL = 100

    resume = args.resume

    logger = get_logger(args.store_dir)
    logger.info(get_args_status_string(parser, args))
    logger.info(f'Device: {device}')

    # create environment
    # env = gym.make("PongNoFrameskip-v4")
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
        state_type=args.state,
    )
    # TODO: consider removing some of the wrappers - may improve performance
    env = make_env(env, stack_frames=not args.random_agent, episodic_life=True, clip_rewards=True)

    # create networks
    architecture = args.network
    pretrain = args.pretrain
    if architecture == 'dqn_pong_model':
        policy_net = DQN(n_actions=env.action_space.n).to(device)
        target_net = DQN(n_actions=env.action_space.n).to(device)
        target_net.load_state_dict(policy_net.state_dict())
    elif architecture == 'dueling_dqn':
        policy_net = DuelingDQN(n_actions=env.action_space.n).to(device)
        target_net = DuelingDQN(n_actions=env.action_space.n).to(device)
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
                    resnet18,
                    resnet10,
                    resnet12,
                    resnet14''')

        num_ftrs = policy_net.fc.in_features
        policy_net.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        policy_net.fc = nn.Linear(num_ftrs, env.action_space.n)
        policy_net = policy_net.to(device)
        num_ftrs = target_net.fc.in_features
        target_net.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        target_net.fc = nn.Linear(num_ftrs, env.action_space.n)
        target_net = target_net.to(device)
        target_net.load_state_dict(policy_net.state_dict())

    # setup optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    steps_done = 0
    epoch = 0

    if resume:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        policy_net.load_state_dict(checkpoint['Net'])
        optimizer.load_state_dict(checkpoint['Optimizer'])
        target_net.load_state_dict(policy_net.state_dict())
        steps_done = checkpoint['Steps_Done']
        epoch = checkpoint['Epoch']
        logger.info("Loading the trained model")
        history = pickle.load(open(args.history, 'rb'))
    else:
        history = []

    # initialize replay memory
    if PRIORITY:
        if args.rankbased:
            memory = PrioritizedReplay(MEMORY_SIZE, rank_based=True)
        else:
            memory = PrioritizedReplay(MEMORY_SIZE)
    else:
        memory = ReplayMemory(MEMORY_SIZE)

    if args.test:  # test
        test(env, 1, policy_net, render=RENDER)
    else:  # train
        history = train(env, args.episodes, history, render=RENDER)
        save_checkpoint(args.store_dir)
