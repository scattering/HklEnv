from gym.envs.registration import registry, register, make, spec

# Algorithmic
# ----------------------------------------
register(
    id='hkl-v0',
    entry_point='gym.envs.crystals:HklEnv',
)
