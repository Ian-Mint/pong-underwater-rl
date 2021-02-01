# DQN PyTorch
This is a simple implementation of the Deep Q-learning algorithm on the Atari Pong environment.

![](/underwater_rl/assets/pong.gif)

## Usage

```shell script
$ python main.py --help
usage: main.py [-h] [--width WIDTH] [--height HEIGHT] [--ball BALL] [--ball-size BALL_SIZE] [--ball-volume] [--snell SNELL [SNELL ...]] [--no-refraction] [--uniform-speed] [--snell-width SNELL_WIDTH] [--snell-change SNELL_CHANGE]
               [--snell-visible {human,machine,none}] [--paddle-speed PADDLE_SPEED] [--paddle-angle PADDLE_ANGLE] [--paddle-length PADDLE_LENGTH] [--update-prob UPDATE_PROB] [--state {binary,color}] [--learning-rate LEARNING_RATE]
               [--network {dqn,soft_dqn,dueling_dqn,resnet18,resnet10,resnet12,resnet14,noisy,lstm,distribution_dqn}] [--double] [--pretrain] [--test] [--save-transitions] [--render {human,png}] [--epsdecay EPSDECAY]
               [--steps-decay] [--episodes EPISODES] [--replay REPLAY] [--priority] [--rank-based] [--batch-size BATCH_SIZE] [--compress] [--actors N_ACTORS] [--samplers N_SAMPLERS] [--resume] [--checkpoint CHECKPOINT]
               [--start-episode START_EPISODE] [--store-dir STORE_DIR] [--debug]

Dynamic Pong RL

optional arguments:
  -h, --help            show this help message and exit

Environment:
  Environment controls

  --width WIDTH         canvas width (default: 160)
  --height HEIGHT       canvas height (default: 160)
  --ball BALL           ball speed (default: 1.0)
  --ball-size BALL_SIZE
                        ball size (default: 2.0)
  --ball-volume         If set, the ball interacts as if it has volume
  --snell SNELL [SNELL ...]
                        snell speed (default: 1.0); or {min, max}, s.t. snell speed is min for actor 1, max for actor `n`, and interpolated between
  --no-refraction       set to disable refraction
  --uniform-speed       set to disable a different ball speed in the Snell layer
  --snell-width SNELL_WIDTH
                        snell speed (default: 80.0)
  --snell-change SNELL_CHANGE
                        Standard deviation of the speed change per step (default: 0)
  --snell-visible {human,machine,none}
                        Determine whether snell is visible to when rendering ('render') or to the agent and when rendering ('machine')
  --paddle-speed PADDLE_SPEED
                        paddle speed (default: 1.0)
  --paddle-angle PADDLE_ANGLE
                        Maximum angle the ball can leave the paddle (default: 70deg)
  --paddle-length PADDLE_LENGTH
                        paddle length (default: 20)
  --update-prob UPDATE_PROB
                        Probability that the opponent moves in the direction of the ball (default: 0.4)
  --state {binary,color}
                        state representation (default: binary)

Model:
  Reinforcement learning model parameters

  --learning-rate LEARNING_RATE
                        learning rate (default: 1e-4)
  --network {dqn,soft_dqn,dueling_dqn,resnet18,resnet10,resnet12,resnet14,noisy,lstm,distribution_dqn}
                        choose a network architecture (default: dqn)
  --double              switch for double dqn (default: False)
  --pretrain            switch for pretrained network (default: False)
  --test                Run the model without training
  --save-transitions    If true, save transitions in "transitions.p"
  --render {human,png}  Rendering mode. Omit if no rendering is desired.
  --epsdecay EPSDECAY   _epsilon decay (default: 1000)
  --steps-decay         switch to use default step decay
  --episodes EPISODES   Number of episodes to train for (default: 4000)
  --replay REPLAY       change the replay mem size (default: 100,000)
  --priority            switch for prioritized replay (default: False)
  --rank-based          switch for rank-based prioritized replay (omit if proportional)

Computation:
  Computational performance parameters

  --batch-size BATCH_SIZE
                        network training batch size or sequence length for recurrent networks
  --compress            If set, store states compressed as png images. Add one CPU if set
  --actors N_ACTORS     Number of actors to use. 3 + n_actors CPUs required
  --samplers N_SAMPLERS
                        Number of sampler processes to use. An equal number of decoder processes will spawn.

Resume:
  Store experiments / Resume training

  --resume              Resume training switch. (omit to start from scratch)
  --checkpoint CHECKPOINT
                        Checkpoint to load if resuming (default: dqn.torch)
  --start-episode START_EPISODE
                        If resuming, restart at this episode (default: 0)
  --store-dir STORE_DIR
                        Path to directory to store experiment results (default: ./experiments/<timestamp>/

Debug:
  --debug               Debug mode
```
