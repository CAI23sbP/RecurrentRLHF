
import collections
from typing import (
    Any,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Sequence,
    Union,
)
import numpy as np

from .recurrent_types import RecurrentTrajectoryWithRew, RecurrentTransitionsWithRew, RecurrentTransitions, RecurrentTrajectory
from imitation.data.rollout import flatten_trajectories
from imitation.data.types import (
    Observation, 
    stack_maybe_dictobs, 
    maybe_wrap_in_dictobs,
    dataclass_quick_asdict ,
    concatenate_maybe_dictobs,
    DictObs,
)

class RecurrentTrajectoryAccumulator:
    def __init__(self):
        self.partial_trajectories = collections.defaultdict(list)

    def add_step(
        self,
        step_dict: Mapping[str, Union[Observation, Mapping[str, Any]]],
        key: Hashable = None,
    ) -> None:
        self.partial_trajectories[key].append(step_dict)

    def finish_trajectory(
        self,
        key: Hashable,
        terminal: bool,
    ) -> RecurrentTrajectoryWithRew:
        part_dicts = self.partial_trajectories[key]
        del self.partial_trajectories[key]
        out_dict_unstacked = collections.defaultdict(list)
        for part_dict in part_dicts:
            for k, array in part_dict.items():
                out_dict_unstacked[k].append(array)

        out_dict_stacked = {
            k: stack_maybe_dictobs(arr_list)
            for k, arr_list in out_dict_unstacked.items()
        }
        traj = RecurrentTrajectoryWithRew(**out_dict_stacked, terminal=terminal)
        assert traj.rews.shape[0] == traj.acts.shape[0] == len(traj.obs) - 1
        return traj

    def add_steps_and_auto_finish(
        self,
        acts: np.ndarray,
        obs: Union[Observation, Dict[str, np.ndarray]],
        rews: np.ndarray,
        dones: np.ndarray,
        infos: List[dict],
        hidden_states: np.ndarray,
    ) -> List[RecurrentTrajectoryWithRew]:
        trajs: List[RecurrentTrajectoryWithRew] = []
        wrapped_obs = maybe_wrap_in_dictobs(obs)

        for env_idx in range(len(wrapped_obs)):
            assert env_idx in self.partial_trajectories
            assert list(self.partial_trajectories[env_idx][0].keys()) == ["obs"], (
                "Need to first initialize partial trajectory using "
                "self._traj_accum.add_step({'obs': ob}, key=env_idx)"
            )
        if len(hidden_states.shape)> 3: ## ensembling recurrent
            hidden_states = hidden_states.swapaxes(0,2) # 3, 1, 8, 120 - > 8 , 3 , 1, 120  -> 8, 1, 3, 120
        else:
            hidden_states = hidden_states.swapaxes(0,1) # 1, 8, 120 -> 8, 1, 120

        zip_iter = enumerate(zip(acts, wrapped_obs, rews, dones, infos, hidden_states))
        for env_idx, (act, ob, rew, done, info, hidden_state) in zip_iter:
            if done:
                real_ob = maybe_wrap_in_dictobs(info["terminal_observation"])
            else:
                real_ob = ob
            self.add_step(
                dict(
                    acts=act,
                    rews=rew,
                    obs=real_ob,
                    infos=info,
                    hidden_states = hidden_state
                ),
                env_idx,
            )
            if done:
                new_traj = self.finish_trajectory(env_idx, terminal=True)
                trajs.append(new_traj)
                self.add_step(dict(obs=ob), env_idx)
        return trajs

def flatten_trajectories(
    trajectories: Iterable[RecurrentTrajectory],
) -> RecurrentTransitions:
    def all_of_type(key, desired_type):
        return all(
            isinstance(getattr(traj, key), desired_type) for traj in trajectories
        )

    assert all_of_type("obs", DictObs) or all_of_type("obs", np.ndarray)
    assert all_of_type("acts", np.ndarray)

    keys = ["obs", "next_obs", "acts", "dones", "infos", "hidden_states"]
    parts: Mapping[str, List[Any]] = {key: [] for key in keys}
    for traj in trajectories:
        parts["acts"].append(traj.acts)

        obs = traj.obs
        parts["obs"].append(obs[:-1])
        parts["next_obs"].append(obs[1:])

        dones = np.zeros(len(traj.acts), dtype=bool)
        dones[-1] = traj.terminal
        parts["dones"].append(dones)

        if traj.infos is None:
            infos = np.array([{}] * len(traj))
        else:
            infos = traj.infos
        parts["infos"].append(infos)

        if len(traj.hidden_states.shape)> 3:
            parts["hidden_states"].append(traj.hidden_states.swapaxes(0,2))
        else:
            parts["hidden_states"].append(traj.hidden_states)


    cat_parts = {
        key: concatenate_maybe_dictobs(part_list)
        for key, part_list in parts.items()
    }
    lengths = set(map(len, cat_parts.values()))
    # assert len(lengths) == 1, f"expected one length, got {lengths}"
    return RecurrentTransitions(**cat_parts)



def flatten_recurrent_trajectories(
    trajectories: Sequence[RecurrentTrajectoryWithRew],
) -> RecurrentTransitionsWithRew:
    transitions = flatten_trajectories(trajectories)
    rews = np.concatenate([traj.rews for traj in trajectories])
    hidden_states = np.concatenate([traj.hidden_states for traj in trajectories])
    return RecurrentTransitionsWithRew(
        **dataclass_quick_asdict(transitions),
        rews=rews,
        hidden_states= hidden_states
    )
