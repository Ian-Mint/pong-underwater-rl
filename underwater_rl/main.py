#!/usr/bin/env python3
"""
Distributed reinforcement learning for gym-dynamic-pong. Requires a Linux operating system.

Memory and sampling pipeline:
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
Learner                                                                          +-> _sample
"""
import argparse
import os
import sys
import time
from typing import Iterable

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

try:
    import underwater_rl
except ImportError:
    sys.path.append(os.path.abspath(os.path.pardir))
if not ('linux' in sys.platform):
    raise Warning(f"{sys.platform} is not supported")

from underwater_rl.actor import Actor, ActorTest, N_ACTIONS
from underwater_rl.common import ParamPipe, DEVICE, Comms
from underwater_rl.learner import Learner
from underwater_rl.models import *
from underwater_rl.utils import *
from underwater_rl.replay import Replay

"""
With one replay_out_queue per decoder
16 decoders:
- 100 samples in 8-10 seconds
- replay_out_queue does not fill

8 decoders:
- 100 samples in 5-6 seconds
- replay_out_queue does not fill

4 decoders:
- 100 samples in 6-7 seconds
- replay_out_queue oscillates
"""


def initialize_model(architecture: str) -> nn.Module:
    r"""
    Return model based on the chosen architecture.

    :param architecture: model selector string
    :return: Selected model object
    """
    model_lookup = {'dqn': DQN,
                    'soft_dqn': softDQN,
                    'dueling_dqn': DuelingDQN,
                    'lstm': DRQN,
                    'distributional_dqn': DistributionalDQN}
    return model_lookup[architecture](n_actions=N_ACTIONS)  # Allow users of the model to put it on the desired device


def get_parser() -> argparse.ArgumentParser:
    """
    Generate command line argument parser

    :return: parser
    """
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
    rl_args.add_argument('--save-transitions', default=False, action='store_true',
                         help='If true, save transitions in "transitions.p"')
    rl_args.add_argument('--render', default=False, type=str, choices=['human', 'png'],
                         help="Rendering mode. Omit if no rendering is desired.")
    rl_args.add_argument('--epsdecay', default=1000, type=int,
                         help="_epsilon decay (default: 1000)")
    rl_args.add_argument('--steps-decay', dest='steps_decay', default=False, action='store_true',
                         help="switch to use default step decay")
    rl_args.add_argument('--episodes', dest='episodes', default=4000, type=int,
                         help='Number of episodes to train for (default: 4000)')
    rl_args.add_argument('--replay', default=100_000, type=int,
                         help="change the replay mem size (default: 100,000)")
    rl_args.add_argument('--priority', default=False, action='store_true',
                         help='switch for prioritized replay (default: False)')
    rl_args.add_argument('--rank-based', dest='rank_based', default=False, action='store_true',
                         help='switch for rank-based prioritized replay (omit if proportional)')

    '''computation args'''
    comp_args = parser.add_argument_group("Computation", "Computational performance parameters")
    comp_args.add_argument('--batch-size', dest='batch_size', default=512, type=int,
                           help="network training batch size or sequence length for recurrent networks")
    comp_args.add_argument('--compress', dest='compress_state', action='store_true', default=False,
                           help="If set, store states compressed as png images. Add one CPU if set")
    comp_args.add_argument('--actors', dest='n_actors', type=int, default=1,
                           help="Number of actors to use. 3 + n_actors CPUs required")
    comp_args.add_argument('--samplers', dest='n_samplers', type=int, default=2,
                           help="Number of sampler processes to use. An equal number of decoder processes will spawn.")

    '''resume args'''
    resume_args = parser.add_argument_group("Resume", "Store experiments / Resume training")
    resume_args.add_argument('--resume', dest='resume', action='store_true',
                             help='Resume training switch. (omit to start from scratch)')
    resume_args.add_argument('--checkpoint', default='dqn.torch',
                             help='Checkpoint to load if resuming (default: dqn.torch)')
    resume_args.add_argument('--history', default='history.p',
                             help='History to load if resuming (default: history.p)')
    resume_args.add_argument('--store-dir', dest='store_dir',
                             default=os.path.join('..', 'experiments', time.strftime("%Y-%m-%d %H.%M.%S")),
                             help='Path to directory to store experiment results (default: ./experiments/<timestamp>/')

    '''debug args'''
    debug_args = parser.add_argument_group("Debug")
    debug_args.add_argument('--debug', action='store_true', help='Debug mode')

    return parser


def create_storage_dir(directory: str) -> None:
    r"""
    Create directory `dir` if it does not exist
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_checkpoint(path: str, model) -> None:
    """
    Load policy net and target_net state from file
    
    :param path: path to the checkpoint 
    :param model: policy model object
    """
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['policy_state'])


def main() -> None:
    """
    Code here is called in the main process, but not by subprocesses.
    """
    parser = get_parser()
    args = parser.parse_args()
    create_storage_dir(args.store_dir)

    logger, log_queue = get_logger(args.store_dir)
    logger.info(get_args_status_string(parser, args))
    logger.info(f'Device: {DEVICE}')

    if args.test:
        test(args, logger)
    else:
        train(args, logger, log_queue)


def test(args, logger):
    actor_params = {
        'test_mode': True,
        'save_transitions': args.save_transitions,
        'architecture': args.network,
        'steps_decay': args.steps_decay,
        'eps_decay': args.epsdecay,
        'eps_end': 0.02,
        'eps_start': 1,
    }

    model = initialize_model(args.network)
    load_checkpoint(args.checkpoint, model)

    actor = ActorTest(model=model,
                      n_episodes=args.episodes,
                      render_mode=args.render,
                      global_args=args,
                      logger=logger,
                      actor_params=actor_params)

    actor.start()

    try:
        actor.join()
        logger.info("All actors finished")
    except KeyboardInterrupt:
        del actor


def train(args, logger, log_queue):
    learning_params = {
        'batch_size': args.batch_size,
        'gamma': 0.99,
        'learning_rate': args.learning_rate,
        'prioritized': args.priority,
        'double': args.double,
        'architecture': args.network,
    }

    # todo: subclass actor to implement steps_decay
    if args.steps_decay:
        raise NotImplementedError("steps_decay is not yet implemented")

    actor_params = {
        'test_mode': False,
        'architecture': args.network,
        'steps_decay': args.steps_decay,
        'eps_decay': args.epsdecay,
        'eps_end': 0.02,
        'eps_start': 1,
    }

    replay_params = {
        'memory_size': args.replay,
        'initial_memory': args.replay,  # Wait for the memory buffer to fill before training
        'batch_size': args.batch_size
    }

    # Get shared objects
    model = initialize_model(args.network)
    comms = get_communication_objects(args.n_actors, args.n_samplers)

    # Create subprocesses
    actors = []
    for p in comms.pipes:
        a = Actor(model=model,
                  n_episodes=args.episodes,
                  render_mode=args.render,
                  memory_queue=comms.memory_q,
                  replay_in_queue=comms.replay_in_q,
                  pipe=p,
                  global_args=args,
                  log_queue=log_queue,
                  actor_params=actor_params)
        actors.append(a)

    learner = Learner(optimizer=optim.Adam,
                      model=model,
                      replay_out_queues=comms.replay_out_q,
                      sample_queue=comms.sample_q,
                      pipes=comms.pipes,
                      checkpoint_path=os.path.join(args.store_dir, 'dqn.torch'),
                      log_queue=log_queue,
                      learning_params=learning_params,
                      n_decoders=args.n_samplers)
    replay = Replay(comms.replay_in_q, comms.replay_out_q, log_queue, replay_params)

    # Start subprocesses
    run_all(actors, 'start')
    replay.start()
    learner.start()

    try:
        # Join subprocess. actor is the only one that is not infinite.
        run_all(actors, 'join')
        logger.info("All actors finished")
    except KeyboardInterrupt:
        run_all(actors, '__del__')
    finally:
        del replay
        del learner


def run_all(actors: Iterable[Actor], method: str, *args, **kwargs) -> None:
    r"""
    Executes `method` for all actors in `actors`. Equivalent to
    ```
    for a in actors:
        a.method(*args, **kwargs)
    ```

    :param actors: Iterable of actor objects
    :param method: Actor.method to execute
    :param args:
    :param kwargs:
    """
    for a in actors:
        a.__getattribute__(method)(*args, **kwargs)


def get_communication_objects(n_pipes: int, n_samplers: int) -> Comms:
    r"""
    Return the various queues and pipes used to communicate between processes

    :param n_pipes: Number of ParamPipes. Should equal the number of actors.
    :param n_samplers: number of sampler processes to use.
    :return: (memory_queue, replay_in_queue, replay_out_queue, sample_queue, pipes)
    """
    memory_queue = mp.Queue(maxsize=1_000)
    replay_in_queue = mp.Queue(maxsize=1_000)
    replay_out_queues = [mp.Queue(maxsize=1_000) for _ in range(n_samplers)]
    sample_queue = mp.Queue(maxsize=20)

    pipes = [ParamPipe() for _ in range(n_pipes)]
    return Comms(memory_queue, replay_in_queue, replay_out_queues, sample_queue, pipes)


if __name__ == '__main__':
    mp.set_start_method('forkserver')  # CUDA is incompatible with 'fork'
    main()
