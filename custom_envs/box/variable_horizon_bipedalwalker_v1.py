
import gymnasium as gym
from gymnasium.envs.box2d.bipedal_walker import BipedalWalker
from typing import Optional

class VHBipedalWalker_v1(BipedalWalker):

    def __init__(self, render_mode: Optional[str] = None, hardcore: bool = False):
        super().__init__(render_mode, hardcore)

    def reset(self, *args, **kwargs):
        return super().reset(*args, **kwargs)

    def step(self, action):

        obs, rew, done, _, info = super().step(action)

        return obs, rew, done, False, info
