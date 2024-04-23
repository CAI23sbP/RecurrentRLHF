
import gymnasium as gym
from gymnasium.envs.box2d.bipedal_walker import BipedalWalker
from typing import Optional

"""
See reference https://github.com/HumanCompatibleAI/seals/blob/master/src/seals/util.py
"""

class FHBipedalWalker_v1(BipedalWalker):

    def __init__(self, render_mode: Optional[str] = None, hardcore: bool = False):
        super().__init__(render_mode, hardcore)
        self.at_absorb_state = False
        self.absorb_obs_this_episode = None
        self.absorb_reward = None 

    def reset(self, *args, **kwargs):
        self.at_absorb_state = False
        self.absorb_obs_this_episode = None
        self.absorb_reward = None 
        return super().reset(*args, **kwargs)

    def step(self, action):

        if not self.at_absorb_state:
            obs, rew, terminated, truncated, info = super().step(action)
            if terminated or truncated:
                self.at_absorb_state = True
                self.absorb_obs_this_episode = obs
                self.absorb_reward = rew
        else:
            obs = self.absorb_obs_this_episode
            rew = 0
            info = {}
            
            if self.render_mode == "human":
                self.render()
                
        return obs, rew, False, False, info

