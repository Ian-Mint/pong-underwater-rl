from gym.envs.registration import register

register(
    id='dynamic-pong-v0',
    entry_point='gym_dynamic_pong.envs:DynamicPongEnv',
    nondeterministic=False,  # TODO: non-deterministic should be True
)

register(
    id='par-dynamic-pong-v0',
    entry_point='gym_dynamic_pong.envs:ParDynamicPongEnv',
    nondeterministic=True,
)
