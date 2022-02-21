from gym.envs.registration import register

register(
    id='AchtungDieKurve-v1',
    entry_point='gym_achtung.envs:AchtungDieKurve',
    max_episode_steps=100000,
    reward_threshold=1.0,
    nondeterministic=True,
)

register(
    id='AchtungDieKurveRandomOpponent-v1',
    entry_point='gym_achtung.envs:AchtungDieKurveRandomOpponent',
    max_episode_steps=100000,
    reward_threshold=1.0,
    nondeterministic=True,
)

register(
    id='AchtungDieKurveFullImage-v1',
    entry_point='gym_achtung.envs:AchtungDieKurveFullImage',
    max_episode_steps=100000,
    reward_threshold=1.0,
    nondeterministic=True,
)

register(
    id='AchtungDieKurveFullImageRandomOpponent-v1',
    entry_point='gym_achtung.envs:AchtungDieKurveFullImageRandomOpponent',
    max_episode_steps=100000,
    reward_threshold=1.0,
    nondeterministic=True,
)

register(
    id='AchtungDieKurveAgainstBot-v1',
    entry_point='gym_achtung.envs:AchtungDieKurveAgainstBot',
    max_episode_steps=100000,
    reward_threshold=1.0,
    nondeterministic=True,
)

register(
    id='AchtungPmf-v1',
    entry_point='gym_achtung.envs:AchtungPmf',
    max_episode_steps=100000,
    reward_threshold=1.0,
    nondeterministic=True,
)
