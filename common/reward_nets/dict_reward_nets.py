import gymnasium  as gym 
import torch 
import torch.nn as nn 
import numpy as np 

from typing import Iterable, Tuple,  cast

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common import preprocessing
from torch import nn

from imitation.util import networks, util
from imitation.rewards.reward_nets import RewardNet, RewardEnsemble

class DictRewardNet(RewardNet):

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        normalize_images: bool = True,
    ):
        
        super().__init__(observation_space ,action_space )

    def preprocess(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,

    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state_th = {key: util.safe_to_tensor(state_).to(self.device) for key, state_ in state.items()}
        action_th = util.safe_to_tensor(action).to(self.device)
        next_state_th = {key: util.safe_to_tensor(next_state_).to(self.device) for key, next_state_ in next_state.items()}
        done_th = util.safe_to_tensor(done).to(self.device)

        del state, action, next_state, done  # unused

        state_th = cast(
                        torch.Tensor,
                        preprocessing.preprocess_obs(
                            state_th,
                            self.observation_space,
                            self.normalize_images,
                        ),
                    ) 
        
        action_th = cast(
            torch.Tensor,
            preprocessing.preprocess_obs(
                action_th,
                self.action_space,
                self.normalize_images,
            ),
        )

        next_state_th = cast(
                        torch.Tensor,
                        preprocessing.preprocess_obs(
                            next_state_th,
                            self.observation_space,
                            self.normalize_images,
                        ),
                    ) 
        done_th = done_th.to(torch.float32)

        
        for key in state_th.keys():
            assert state_th[key].shape == next_state_th[key].shape
            n_gen = len(state_th[key])
        assert len(action_th) == n_gen

        return state_th, action_th, next_state_th, done_th

    def predict_th(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> torch.Tensor:
        with networks.evaluating(self):

            state_th, action_th, next_state_th, done_th = self.preprocess(
                state,
                action,
                next_state,
                done,
            )
            with torch.no_grad():
                rew_th = self(state_th, action_th, next_state_th, done_th)

            for state_ in state_th.values(): 
                assert rew_th.shape == state_.shape[:1]
            return rew_th

    def predict(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> np.ndarray:
        rew_th = self.predict_th(state, action, next_state, done)
        return rew_th.detach().cpu().numpy().flatten()

    def predict_processed(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        
        del kwargs
        return self.predict(state, action, next_state, done)


class DictRewardEnsemble(RewardEnsemble):


    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        members: Iterable[RewardNet],
    ):
        super().__init__(observation_space, action_space, members)

    def predict_processed_all(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        for key in state.keys():
            batch_size = state.shape[key][0]
        rewards_list = [
            member.predict_processed(state, action, next_state, done, **kwargs)
            for member in self.members
        ]
        rewards: np.ndarray = np.stack(rewards_list, axis=-1)
        assert rewards.shape == (batch_size, self.num_members)
        return rewards

    @th.no_grad()
    def predict_reward_moments(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        for key in state.keys():
            batch_size = state.shape[key][0]
        all_rewards = self.predict_processed_all(
            state,
            action,
            next_state,
            done,
            **kwargs,
        )
        mean_reward = all_rewards.mean(-1)
        var_reward = all_rewards.var(-1, ddof=1)
        assert mean_reward.shape == var_reward.shape == (batch_size,)
        return mean_reward, var_reward
