
import gymnasium as gym
from gymnasium.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
from typing import Optional
"""
See reference https://github.com/HumanCompatibleAI/seals/blob/master/src/seals/util.py
"""
class FHContinuous_MountainCarEnv(Continuous_MountainCarEnv):

    def __init__(self, render_mode: Optional[str] = None, goal_velocity: int = 0):
        super().__init__(render_mode, goal_velocity)
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
            rew = self.absorb_reward
            info = {}
            
            if self.render_mode == "human":
                self.render()
                
        return obs, rew, False, False, info
