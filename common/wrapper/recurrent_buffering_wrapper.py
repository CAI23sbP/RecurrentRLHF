
from typing import List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from stable_baselines3.common.vec_env import VecEnv

# from imitation.data import rollout, types
from ..data import recurrent_rollout, recurrent_types
from imitation.data.wrappers import BufferingWrapper
from imitation.rewards import reward_nets
import torch as th 
from copy import deepcopy
from imitation.data.types import maybe_unwrap_dictobs, maybe_wrap_in_dictobs, DictObs

import collections
from typing import Deque
from imitation.rewards.reward_wrapper import WrappedRewardCallback


class RecurrentBufferingWrapper(BufferingWrapper):

    """
    merge with BufferingWrapper and RewardVecEnvWrapper 
    for recurrenting Hidden_state
    """

    error_on_premature_event: bool
    _trajectories: List[recurrent_types.RecurrentTrajectoryWithRew]
    _ep_lens: List[int]
    _init_reset: bool
    _traj_accum: Optional[recurrent_rollout.RecurrentTrajectoryAccumulator]
    _timesteps: Optional[npt.NDArray[np.int_]]
    n_transitions: Optional[int]

    def __init__(self, 
                 venv: VecEnv, 
                 reward_fn: reward_nets.RewardNet,
                 error_on_premature_reset: bool = True,
                 ep_history: int = 100,
                 member: int = 1):
        super().__init__(venv, error_on_premature_reset)
        if member == 1:
            single_hidden_state_shape = (reward_fn.gru.num_layers, self.venv.num_envs, reward_fn.gru.hidden_size)
        else:
            single_hidden_state_shape = (member, reward_fn.gru.num_layers, self.venv.num_envs, reward_fn.gru.hidden_size)
        _last_states = th.zeros(single_hidden_state_shape, device=reward_fn.device)
        self.reward_fn = reward_fn.predict_processed
        self.hidden_states = deepcopy(_last_states)
        self._old_obs = None
        self.episode_rewards: Deque = collections.deque(maxlen=ep_history)
        self._cumulative_rew = np.zeros((venv.num_envs,))
        self.reset()

    def make_log_callback(self) -> WrappedRewardCallback:
        return WrappedRewardCallback(self.episode_rewards)

    @property
    def envs(self):
        return self.venv.envs

    def reset(self, **kwargs):
        if (
            self._init_reset
            and self.error_on_premature_reset
            and self.n_transitions > 0
        ):  # noqa: E127
            raise RuntimeError("BufferingWrapper reset() before samples were accessed")
        self._init_reset = True
        self.n_transitions = 0
        obs = self.venv.reset(**kwargs)
        self._traj_accum = recurrent_rollout.RecurrentTrajectoryAccumulator()
        obs = maybe_wrap_in_dictobs(obs)
        for i, ob in enumerate(obs):
            self._traj_accum.add_step({"obs": ob}, key=i)
        self._timesteps = np.zeros((len(obs),), dtype=int)
        obs = maybe_unwrap_dictobs(obs)
        self._old_obs = obs
        return obs

    def step_async(self, actions):
        assert self._init_reset
        assert self._saved_acts is None
        self._saved_acts = actions
        self.venv.step_async(actions)
        

    def step_wait(self):
        assert self._init_reset
        assert self._saved_acts is not None
        acts, self._saved_acts = self._saved_acts, None
        obs_fixed = []
        obs, old_rews, dones, infos = self.venv.step_wait()
        for single_obs, single_done, single_infos in zip(obs, dones, infos):
            if single_done:
                single_obs = single_infos["terminal_observation"]

            obs_fixed.append(maybe_wrap_in_dictobs(single_obs))
        obs_fixed = (
            DictObs.stack(obs_fixed)
            if isinstance(obs, DictObs)
            else np.stack(obs_fixed)
        )

        self.n_transitions += self.num_envs
        self._timesteps += 1
        ep_lens = self._timesteps[dones]
        if len(ep_lens) > 0:
            self._ep_lens += list(ep_lens)
        self._timesteps[dones] = 0

        rews, self.hidden_states = self.reward_fn(self._old_obs , 
                                               acts,  
                                               maybe_unwrap_dictobs(obs_fixed),
                                               np.array(dones), 
                                               self.hidden_states)
        
        finished_trajs = self._traj_accum.add_steps_and_auto_finish(
            acts,
            obs,
            old_rews,
            dones,
            infos,
            self.hidden_states
        )
        
        self._trajectories.extend(finished_trajs)
        assert len(rews) == len(obs), "must return one rew for each env"
        done_mask = np.asarray(dones, dtype="bool").reshape((len(dones),))

        self._cumulative_rew += rews
        for single_done, single_ep_rew in zip(dones, self._cumulative_rew):
            if single_done:
                self.episode_rewards.append(single_ep_rew)
        self._cumulative_rew[done_mask] = 0
        obs = maybe_unwrap_dictobs(obs)
        self._old_obs = obs
        for info_dict, old_rew in zip(infos, old_rews):
            info_dict["original_env_rew"] = old_rew
        return obs, rews, dones, infos


    def _finish_partial_trajectories(self) -> Sequence[recurrent_types.RecurrentTrajectoryWithRew]:
        """Finishes and returns partial trajectories in `self._traj_accum`."""
        assert self._traj_accum is not None
        trajs = []
        for i in range(self.num_envs):
            n_transitions = len(self._traj_accum.partial_trajectories[i]) - 1
            assert n_transitions >= 0, "Invalid TrajectoryAccumulator state"
            if n_transitions >= 1:
                traj = self._traj_accum.finish_trajectory(i, terminal=False)
                trajs.append(traj)
                self._traj_accum.add_step({"obs": traj.obs[-1]}, key=i)
        return trajs

    def pop_finished_trajectories(
        self,
    ) -> Tuple[Sequence[recurrent_types.RecurrentTrajectoryWithRew], Sequence[int]]:
        trajectories = self._trajectories
        ep_lens = self._ep_lens
        self._trajectories = []
        self._ep_lens = []
        self.n_transitions = 0
        return trajectories, ep_lens

    def pop_trajectories(
        self,
    ) -> Tuple[Sequence[recurrent_types.RecurrentTransitionsWithRew], Sequence[int]]:
        if self.n_transitions == 0:
            return [], []
        partial_trajs = self._finish_partial_trajectories()
        self._trajectories.extend(partial_trajs)
        return self.pop_finished_trajectories()

    def pop_transitions(self) -> recurrent_types.RecurrentTransitionsWithRew:
        if self.n_transitions == 0:
            raise RuntimeError("Called pop_transitions on an empty BufferingWrapper")
        n_transitions = self.n_transitions
        trajectories, _ = self.pop_trajectories()
        transitions = recurrent_rollout.flatten_recurrent_trajectories(trajectories)
        assert len(transitions.obs) == n_transitions
        return transitions