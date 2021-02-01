# DQN PyTorch
This is a simple implementation of the Deep Q-learning algorithm on the Atari Pong environment.

![](/underwater_rl/assets/pong.gif)

## Usage

```shell script
$ python main.py --help
usage: main.py [-h] [--width WIDTH] [--height HEIGHT] [--ball BALL] [--ball-size BALL_SIZE] [--ball-volume] [--snell SNELL] [--no-refraction] [--uniform-speed] [--snell-width SNELL_WIDTH] [--snell-change SNELL_CHANGE]
               [--snell-visible {human,machine,none}] [--paddle-speed PADDLE_SPEED] [--paddle-angle PADDLE_ANGLE] [--paddle-length PADDLE_LENGTH] [--update-prob UPDATE_PROB] [--state {binary,color}] [--learning-rate LEARNING_RATE]
               [--network {dqn_pong_model,soft_dqn,dueling_dqn,resnet18,resnet10,resnet12,resnet14,noisy_dqn,predict_dqn,lstm,distribution_dqn,attention_dqn}] [--double] [--pretrain] [--test] [--render {human,png}]
               [--epsdecay EPSDECAY] [--stepsdecay] [--episodes EPISODES] [--replay REPLAY] [--priority] [--rankbased] [--batch-size BATCH_SIZE] [--train-prediction] [--pred-episode PRED_EPISODE] [--resume]
               [--checkpoint CHECKPOINT] [--history HISTORY] [--store-dir STORE_DIR] [--debug]

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
  --snell SNELL         snell speed (default: 1.0)
  --no-refraction       set to disable refraction
  --uniform-speed       set to disable a different ball speed in the Snell layer
  --snell-width SNELL_WIDTH
                        width of the snell layer (default: 80.0)
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
  --network {dqn_pong_model,soft_dqn,dueling_dqn,resnet18,resnet10,resnet12,resnet14,noisy_dqn,predict_dqn,lstm,distribution_dqn,attention_dqn}
                        choose a network architecture (default: dqn_pong_model)
  --double              switch for double dqn (default: False)
  --pretrain            switch for pretrained network (default: False)
  --test                Run the model without training
  --render {human,png}  Rendering mode. Omit if no rendering is desired.
  --epsdecay EPSDECAY   epsilon decay (default: 1000)
  --stepsdecay          switch to use default step decay
  --episodes EPISODES   Number of episodes to train for (default: 4000)
  --replay REPLAY       change the replay mem size (default: 100,000)
  --priority            switch for prioritized replay (default: False)
  --rankbased           switch for rank-based prioritized replay (omit if proportional)
  --batch-size BATCH_SIZE
                        network training batch size or sequence length for recurrent networks
  --train-prediction    train prediction(default: False)
  --pred-episode PRED_EPISODE
                        when to start training prediction model

Resume:
  Store experiments / Resume training

  --resume              Resume training switch. (omit to start from scratch)
  --checkpoint CHECKPOINT
                        Checkpoint to load if resuming (default: dqn_pong_model)
  --history HISTORY     History to load if resuming (default: history.p)
  --store-dir STORE_DIR
                        Path to directory to store experiment results (default: ./experiments/<timestamp>/

Debug:
  --debug               Debug mode

```
