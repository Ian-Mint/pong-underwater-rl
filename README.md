# Reinforcement Learning for Underwater Communication

![](/underwater_rl/assets/pong.gif)

Neural networks suffer from the phenomenon of catastrophic 
forgetting which hinders continual or lifelong learning
and leads to erasure of previously acquired skills. In
this paper, we explore whether catastrophic forgetting can
be addressed in a dynamic environment within reinforcement
learning context. Our work is intended as a proof of
concept for underwater communication in which we control
the movement of the receiver present underwater to maximize
the quality of the received audio signal. In particular,
we find that a distributed noisy network can mitigate
within task forgetting by learning a more generalizable policy
which can quickly adapt to changes in the environment.
However, the issue of catastrophic forgetting is still unresolved.
See the [paper](https://github.com/Ian-Mint/pong-underwater-rl/blob/master/paper.pdf)
for further details.

## Structure

```
.
├── dashboard
├── experiments
├── grid-search
├── gym-dynamic-pong
├── scripts
└── underwater_rl
```

The project is broken into four main parts:
1. `dashboard` - experiment dashboard using [dash](https://plotly.com/dash/).
Reads results from `experiments` and `grid-search`. 
See [readme](https://github.com/Ian-Mint/pong-underwater-rl/blob/master/dashboard/README.md).
2. `gym-dynamic-pong` - a custom [OpenAI Gym](https://gym.openai.com/) environment.
See [readme](https://github.com/Ian-Mint/pong-underwater-rl/blob/master/gym-dynamic-pong/README.md)
3. `scripts` - scripts and tools to launch experiments using [Kubernetes](https://kubernetes.io/).
See [readme](https://github.com/Ian-Mint/pong-underwater-rl/blob/master/scripts/README.md).
4. `underwater_rl` - The main code for testing and training reinforcement learning on the `gym-dynamic-pong` environment.
See [main readme](https://github.com/Ian-Mint/pong-underwater-rl/blob/master/underwater_rl/README.md), or 
[distributed readme](https://github.com/Ian-Mint/pong-underwater-rl/blob/distributed/underwater_rl/README.md)

# Branches

Most of the development work is on the `main` branch including all of the model implementations and our attempts at 
leveraging frame prediction. 

The reader should also take a look at the `distributed` branch, which contains all of the distributed learning code.

# Installation.

Install one of the [conda](https://docs.conda.io/en/latest/miniconda.html) environments:

```shell script
conda create env -f [file]
```

Where `[file]` is one of the `yml` files in the root.

Next, install the gym environment:

```shell script
pip install -e path_to/gym-dynamic-pong/
```
