import gymnasium.envs.registration   as  registration

registration.register(
    id='FHBipedalWalker-v1',
    entry_point='custom_envs.box:FHBipedalWalker_v1',
)

registration.register(
    id='FHBipedalWalker-v2',
    entry_point='custom_envs.box:FHBipedalWalker_v2',
)

registration.register(
    id='FHContinuous_MountainCarEnv-v0',
    entry_point='custom_envs.classic_control:FHContinuous_MountainCarEnv',
)

registration.register(
    id = 'VHBipedalWalker-v1',
    entry_point='custom_envs.box:VHBipedalWalker_v1'
)