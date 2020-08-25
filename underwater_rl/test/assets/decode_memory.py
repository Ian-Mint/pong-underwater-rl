"""
Run this script to produce decoded-memory.p given memory.p
"""

import pickle
import multiprocessing as mp
import time

from underwater_rl.base import Transition
from underwater_rl.learner import Decoder
from underwater_rl.main import get_logger

BATCH_SIZE = 512

logger, log_queue = get_logger('../tmp')

with open('memory-np.p', 'rb') as f:
    memory = pickle.load(f)

replay_out_queue = mp.Queue()
sample_queue = mp.Queue()
batch = []
for i in range(BATCH_SIZE):
    m = memory[i]
    assert isinstance(m, (tuple, list))
    batch.append(Transition(*m))
replay_out_queue.put(batch)

decoder = Decoder(log_queue, replay_out_queue, sample_queue, 0, daemon=True)
decoder.start()
time.sleep(1)

decoded_memory = []
while not sample_queue.empty():
    decoded_memory.append(sample_queue.get())

with open('decoded_memory.p', 'wb') as f:
    pickle.dump(decoded_memory, f)
