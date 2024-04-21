import gymnasium  as gym 
import torch as th 
import numpy as np 
import torch.nn as nn 
from typing import  cast, Tuple, Iterable, Type

import gymnasium as gym
import numpy as np
from stable_baselines3.common import preprocessing

from imitation.util import networks, util
from imitation.rewards.reward_nets import RewardNet, RewardEnsemble, RewardNetWrapper
from .dict_reward_nets import DictRewardNet
from stable_baselines3.common.utils import zip_strict

class RecurrentRewardNet(RewardNet):

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        normalize_images: bool = True,
    ):
        
        super().__init__(observation_space ,action_space )

    @staticmethod
    def _process_sequence(
        features: th.Tensor,
        hidden_state: th.Tensor,
        episode_starts: th.Tensor,
        gru: nn.GRU,
    ) -> Tuple[th.Tensor, th.Tensor]:
        n_seq = hidden_state.shape[1]
        features_sequence = features.reshape((n_seq, -1, gru.input_size)).swapaxes(0, 1)
        episode_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)

        if th.all(episode_starts == 0.0):
            gru_output, hidden_state = gru(features_sequence, hidden_state)
            gru_output = th.flatten(gru_output.transpose(0, 1), start_dim=0, end_dim=1)
            return gru_output, hidden_state
        
        gru_output = []
        for features, episode_start in zip_strict(features_sequence, episode_starts):
            gru_hidden, hidden_state = gru(features.unsqueeze(dim=0), (1.0 - episode_start).view(1, n_seq, 1) * hidden_state)
            gru_output += [gru_hidden]
        # Sequence to batch
        gru_output = th.flatten(th.cat(gru_output).transpose(0, 1), start_dim=0, end_dim=1)
        return gru_output, hidden_state
    
    def preprocess(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        hidden_state: np.ndarray

    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        state_th = util.safe_to_tensor(state).to(self.device)
        action_th = util.safe_to_tensor(action).to(self.device)
        next_state_th = util.safe_to_tensor(next_state).to(self.device)
        done_th = util.safe_to_tensor(done).to(self.device)
        hidden_th = util.safe_to_tensor(hidden_state).to(self.device)

        del state, action, next_state, done, hidden_state  # unused

        state_th = cast(
                        th.Tensor,
                        preprocessing.preprocess_obs(
                            state_th,
                            self.observation_space,
                            self.normalize_images,
                        ),
                    ) 
        
        action_th = cast(
            th.Tensor,
            preprocessing.preprocess_obs(
                action_th,
                self.action_space,
                self.normalize_images,
            ),
        )

        next_state_th = cast(
                        th.Tensor,
                        preprocessing.preprocess_obs(
                            next_state_th,
                            self.observation_space,
                            self.normalize_images,
                        ),
                    ) 
        done_th = done_th.to(th.float32)
        
        hidden_th = hidden_th.to(th.float32)
        n_gen = len(state_th)
        assert state_th.shape == next_state_th.shape
        assert len(action_th) == n_gen

        return state_th, action_th, next_state_th, done_th, hidden_th

    def predict_th(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        hidden_state: np.ndarray
    ) -> Tuple[th.Tensor, th.Tensor]:
        with networks.evaluating(self):

            state_th, action_th, next_state_th, done_th, hidden_th = self.preprocess(
                state,
                action,
                next_state,
                done,
                hidden_state
            )
            with th.no_grad():
                rew_th, hidden_th = self(state_th, action_th, next_state_th, done_th, hidden_th)
                
            assert rew_th.shape == state.shape[:1]
            return rew_th, hidden_th

    def predict(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        hidden_state: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        rew_th, hidden_th = self.predict_th(state, action, next_state, done, hidden_state)
        return rew_th.detach().cpu().numpy().flatten(), hidden_th.detach().cpu().numpy()

    def predict_processed(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        hidden_state: np.ndarray,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        del kwargs
        return self.predict(state, action, next_state, done, hidden_state)


class DictRecurrentRewardNet(RecurrentRewardNet, DictRewardNet):

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        normalize_images: bool = True,
    ):
        
        RecurrentRewardNet.__init__(observation_space ,action_space )
        DictRewardNet.__init__(observation_space ,action_space)

    def preprocess(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        hidden_state: np.ndarray

    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        
        hidden_th = util.safe_to_tensor(hidden_state).to(self.device)
        del hidden_state
        hidden_th = hidden_th.to(th.float32)
        state_th, action_th, next_state_th, done_th = DictRewardNet.preprocess(state, action, next_state, done)
        
        return state_th, action_th, next_state_th, done_th, hidden_th
    
    def predict_th(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        hidden_state: np.ndarray
    ) -> Tuple[th.Tensor, th.Tensor]:
        with networks.evaluating(self):

            state_th, action_th, next_state_th, done_th, hidden_th = self.preprocess(
                state,
                action,
                next_state,
                done,
                hidden_state
            )
            with th.no_grad():
                rew_th, hidden_th = self(state_th, action_th, next_state_th, done_th, hidden_th)

            for state_ in state_th.values(): 
                assert rew_th.shape == state_.shape[:1]
            return rew_th, hidden_th
        
    def predict(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        hidden_state: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        rew_th, hidden_th = self.predict_th(state, action, next_state, done, hidden_state)
        return rew_th.detach().cpu().numpy().flatten(), hidden_th.detach().cpu().numpy()

    def predict_processed(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        hidden_state: np.ndarray,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        del kwargs
        return self.predict(state, action, next_state, done, hidden_state)

import abc 

class RecurerntRewardNetWrapper(RewardNetWrapper):
    def preprocess(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        hidden_state: np.ndarray, 
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        __doc__ = super().preprocess.__doc__  # noqa: F841
        return self.base.preprocess(state, action, next_state, done, hidden_state)

class RecurrentPredictProcessedWrapper(RecurerntRewardNetWrapper):

    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        hidden_state: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor]:
        return self.base.forward(state, action, next_state, done, hidden_state)
   
    @abc.abstractmethod
    def predict_processed(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        hidden_state: np.ndarray, 
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict processed must be overridden in subclasses."""


    def predict(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        hidden_state: np.ndarray, 
    ) -> Tuple[np.ndarray, np.ndarray]:
        __doc__ = super().predict.__doc__  # noqa: F841
        return self.base.predict(state, action, next_state, done, hidden_state)

    def predict_th(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        hidden_state: np.ndarray, 
    ) -> Tuple[th.Tensor, th.Tensor]:
        __doc__ = super().predict_th.__doc__  # noqa: F841
        return self.base.predict_th(state, action, next_state, done, hidden_state)

class RecurrentNormalizedRewardNet(RecurrentPredictProcessedWrapper):

    def __init__(
        self,
        base: RewardNet,
        normalize_output_layer: Type[networks.BaseNorm],
    ):
        super().__init__(base=base)
        self.normalize_output_layer = normalize_output_layer(1)
        self.gru = base.gru 
        self.update_stats = True 

    def predict_processed(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        hidden_state:np.ndarray,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        with networks.evaluating(self):
            rew, hidden_state = self.base.predict_processed(state, action, next_state, done, hidden_state, **kwargs)
            rew_th = th.tensor(rew, device=self.device)
            hidden_th = th.tensor(hidden_state, device=self.device)
            rew = self.normalize_output_layer(rew_th).detach().cpu().numpy().flatten()
        if self.update_stats:
            with th.no_grad():
                self.normalize_output_layer.update_stats(rew_th)
        assert rew.shape == state.shape[:1]
        return rew, hidden_th


## TODO add recurrent RewardEnsemble
class RecurrentRewardEnsemble(RewardEnsemble):
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
        hidden_states: np.ndarray, 
        **kwargs,
    ) -> np.ndarray:
        batch_size = state.shape[0]
        rewards_list = []
        hidden_state_list = []
        for (member, hidden_state) in zip(self.members, hidden_states):
            single_reward, single_hidden_state = member.predict_processed(state, action, next_state, done, hidden_state, **kwargs)
            rewards_list.append(single_reward)
            hidden_state_list.append(single_hidden_state)

        rewards: np.ndarray = np.stack(rewards_list, axis=-1)
        hidden_states: np.ndarray = np.stack(hidden_state_list, axis= 0)

        assert rewards.shape == (batch_size, self.num_members)
        return rewards, hidden_states
    
    @th.no_grad()
    def predict_reward_moments(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        hidden_state: np.ndarray,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        batch_size = state.shape[0]
        all_rewards, all_hiddn_states = self.predict_processed_all(
            state,
            action,
            next_state,
            done,
            hidden_state,
            **kwargs,
        )
        mean_reward = all_rewards.mean(-1)
        var_reward = all_rewards.var(-1, ddof=1)
        assert mean_reward.shape == var_reward.shape == (batch_size,)
        return mean_reward, var_reward, all_hiddn_states

    def predict_processed(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        hidden_state:np.ndarray,
        **kwargs,
    ) -> Tuple[np.ndarray , np.ndarray]:
        return self.predict(state, action, next_state, done, hidden_state,**kwargs)

    def predict(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        hidden_state:np.ndarray,
        **kwargs,
    ) -> Tuple[np.ndarray , np.ndarray] :
        mean, _, hidden_states = self.predict_reward_moments(state, action, next_state, done, hidden_state, **kwargs)
        return mean, hidden_states

# class DictRecurrentRewardEnsemble(RecurrentRewardEnsemble):    
#     def predict_processed_all(
#         self,
#         state: np.ndarray,
#         action: np.ndarray,
#         next_state: np.ndarray,
#         done: np.ndarray,
#         hidden_state: np.ndarray, 
#         **kwargs,
#     ) -> np.ndarray:

#         for key in state.keys():
#             batch_size = state.shape[key][0]
#         rewards_list = []
#         hidden_state_list = []
#         for member in self.members:
#             single_reward, single_hidden_state = member.predict_processed(state, action, next_state, done, hidden_state, **kwargs)
#             rewards_list.append(single_reward)
#             hidden_state_list.append(single_hidden_state)

#         rewards: np.ndarray = np.stack(rewards_list, axis=-1)

#         assert rewards.shape == (batch_size, self.num_members)
#         return rewards, hidden_states

#     @th.no_grad()
#     def predict_reward_moments(
#         self,
#         state: np.ndarray,
#         action: np.ndarray,
#         next_state: np.ndarray,
#         done: np.ndarray,
#         hidden_state: np.ndarray,
#         **kwargs,
#     ) -> Tuple[np.ndarray, np.ndarray]:

#         for key in state.keys():
#             batch_size = state.shape[key][0]

#         all_rewards = self.predict_processed_all(
#             state,
#             action,
#             next_state,
#             done,
#             hidden_state,
#             **kwargs,
#         )
#         mean_reward = all_rewards.mean(-1)
#         var_reward = all_rewards.var(-1, ddof=1)
#         assert mean_reward.shape == var_reward.shape == (batch_size,)
#         return mean_reward, var_reward