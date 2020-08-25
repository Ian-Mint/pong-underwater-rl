"""
Results:
>>> timeit(lambda: learner._policy_state_dict_to_params(), number=10_000)  # copy=True
6.429219907993684

>>> timeit(lambda: learner._policy_state_dict_to_params(), number=10_000)  # copy=False
0.2709134070028085

>>> timeit(lambda: learner._policy_state_dict_to_params(), number=10_000)  # copy=True, non_blocking=True
6.416077349989791

On 4-core CPU:
    190 samples processed in 60 seconds

On GPU:
    2637-2718 samples processed in 60 seconds

Learner runs at 53% GPU given a full queue
"""
from itertools import count
import os
import pickle
import threading
import time
from timeit import timeit
import sys

from torch import optim
import torch.multiprocessing as mp

sys.path.append(os.path.join(os.path.pardir, os.path.pardir))

import underwater_rl.main as rl_main
import underwater_rl.learner as learn

LOG_DIR = 'tmp'
NETWORK = 'dqn'
counter = 0


def queue_maintainer(q: mp.Queue, timeout: int):
    """
    keeps the sample queue full, and counts the number of samples put to the queue to measure the processing rate.
    :return:
    """
    global counter
    with open('assets/decoded_memory.p', 'rb') as f:
        processed_batch = pickle.load(f)

    t_start = time.time()
    while time.time() - t_start > timeout:
        counter += 1
        q.put(processed_batch)

    q.put(None)


def main():
    learning_params = {
        'batch_size': 512,
        'gamma': 0.99,
        'learning_rate': 0e-4,
        'prioritized': False,
        'double': False,
        'architecture': NETWORK,
    }

    logger, log_queue = rl_main.get_logger(LOG_DIR)
    model = rl_main.initialize_model(NETWORK)
    memory_queue, replay_in_queue, replay_out_queue, sample_queue, pipes = rl_main.get_communication_objects(10)
    learner = learn.Learner(
        optimizer=optim.Adam, model=model, replay_out_queue=replay_out_queue, sample_queue=sample_queue,
        pipes=pipes, checkpoint_path=os.path.join(LOG_DIR, 'dqn.torch'), log_queue=log_queue,
        learning_params=learning_params, n_decoders=2, run_profile=True
    )

    test_duration = 60
    queue_pusher = threading.Thread(target=queue_maintainer, args=(sample_queue, test_duration), daemon=True)
    queue_pusher.start()
    time.sleep(1)  # allow the queue to fill

    learner.start()
    time.sleep(test_duration + 20)

if __name__ == '__main__':
    mp.set_start_method('forkserver')
    main()

