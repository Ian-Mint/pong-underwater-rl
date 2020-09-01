import pickle
import os
from queue import Empty
import sys

import torch.multiprocessing as mp

try:
    import underwater_rl
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.pardir, os.pardir, os.pardir)))
from underwater_rl.actor import Actor
from underwater_rl.main import get_communication_objects, initialize_model, get_logger

if __name__ == '__main__':
    mp.set_start_method('forkserver')
    with open('args.p', 'rb') as f:
        args = pickle.load(f)

    actor_params = {
        'test_mode': False,
        'architecture': args.network,
        'steps_decay': args.steps_decay,
        'eps_decay': args.epsdecay,
        'eps_end': 0.02,
        'eps_start': 1,
    }
    logger, log_q = get_logger('../tmp')
    model = initialize_model('dqn')
    comms = get_communication_objects(1)
    actor = Actor(model=model, n_episodes=1, render_mode='none', memory_queue=comms.memory_q,
                  replay_in_queue=comms.replay_in_q, model_params=model_params, global_args=args, log_queue=log_q,
                  actor_params=actor_params)
    actor.start()

    samples = []
    while True:
        try:
            s = comms.memory_q.get(timeout=2)
            samples.append(s)
        except Empty:
            break

    del actor
    with open('raw_samples.p', 'wb') as f:
        pickle.dump(samples, f)
