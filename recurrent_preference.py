from stable_baselines3.common import base_class, type_aliases, utils, vec_env
from stable_baselines3.common.type_aliases import MaybeCallback

from typing import Sequence, Any, Optional, Union, Tuple, cast, NoReturn, overload
from .common.data.recurrent_types import RecurrentTrajectoryWithRew , RecurrentTrajectoryWithRewPair
from .common.wrappers.recurrent_buffering_wrapper import RecurrentBufferingWrapper
from .common.data.recurrent_rollout import * 
from .common.data.recurrent_types import RecurrentTrajectoryPair, RecurrentTransitions
from .common.reward_nets import recurrent_reward_nets
from imitation.util import util
from imitation.data.types import AnyPath, Pair, assert_not_dictobs
from imitation.regularization import regularizers
from imitation.policies import exploration_wrapper
from imitation.util import logger as imit_logger
from imitation.data.rollout import flatten_trajectories, make_sample_until, generate_trajectories, discounted_sum
from imitation.algorithms import preference_comparisons  
from imitation.rewards import reward_function, reward_nets

from scipy import special
from tqdm.auto import tqdm
import torch as th 
import torch.nn as nn 
import pickle, re, math 
import numpy as np 

from torch.utils import data as data_th
from collections import defaultdict

class RecurrentTrajectoryGenerator(preference_comparisons.TrajectoryGenerator):
    def sample(self, steps: int) -> Sequence[RecurrentTrajectoryWithRew]:
        """Sample a batch of trajectories.
        """  # noqa: DAR202

class RecurrentTrajectoryDataset(RecurrentTrajectoryGenerator):
    def __init__(
        self,
        trajectories: Sequence[RecurrentTrajectoryWithRew],
        rng: np.random.Generator,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ):
        super().__init__(custom_logger=custom_logger)
        self._trajectories = trajectories
        self.rng = rng

    def sample(self, steps: int) -> Sequence[RecurrentTrajectoryWithRew]:
        trajectories = list(self._trajectories)
        self.rng.shuffle(trajectories)  # type: ignore[arg-type]
        return _get_trajectories(trajectories, steps)


class RecurrentAgentTrainer(RecurrentTrajectoryGenerator):
    def __init__(
        self,
        algorithm: base_class.BaseAlgorithm,
        reward_fn: Union[reward_function.RewardFn, reward_nets.RewardNet],
        venv: vec_env.VecEnv,
        rng: np.random.Generator,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        exploration_frac: float = 0.0,
        switch_prob: float = 0.5,
        random_prob: float = 0.5,
    ) -> None:
        self.algorithm = algorithm
        super().__init__(custom_logger)
        if isinstance(reward_fn, reward_nets.RewardNet):
            utils.check_for_correct_spaces(
                venv,
                reward_fn.observation_space,
                reward_fn.action_space,
            )
        self.rng = rng
        try:
            member = len(reward_fn.members)
        except AttributeError as e: 
            member = 1 

        self.buffering_wrapper_with_reward_wrapper = RecurrentBufferingWrapper(
            venv = venv,
            reward_fn = reward_fn,
            member = member
        )
        self.log_callback = self.buffering_wrapper_with_reward_wrapper.make_log_callback()
        self.algorithm.set_env(self.buffering_wrapper_with_reward_wrapper)
        algo_venv = self.algorithm.get_env()
        
        self.exploration_wrapper = exploration_wrapper.ExplorationWrapper(
            policy=self.algorithm,
            venv=algo_venv,
            random_prob=random_prob,
            switch_prob=switch_prob,
            rng=self.rng,
        )
        self.exploration_frac = exploration_frac

        assert algo_venv is not None

    def train(self, steps: int, **kwargs) -> None:
        n_transitions = self.buffering_wrapper_with_reward_wrapper.n_transitions
        if n_transitions:
            raise RuntimeError(
                f"There are {n_transitions} transitions left in the buffer. "
                "Call AgentTrainer.sample() first to clear them.",
            )
        self.algorithm.learn(
            total_timesteps=steps,
            reset_num_timesteps=False,
            callback=self.log_callback,
            **kwargs,
        )

    def sample(self, steps: int) -> Sequence[RecurrentTrajectoryWithRew]:
        agent_trajs, _ = self.buffering_wrapper_with_reward_wrapper.pop_finished_trajectories()
        agent_trajs = agent_trajs[::-1]
        avail_steps = sum(len(traj) for traj in agent_trajs)


        exploration_steps = int(self.exploration_frac * steps)
        if self.exploration_frac > 0 and exploration_steps == 0:
            self.logger.warn(
                "No exploration steps included: exploration_frac = "
                f"{self.exploration_frac} > 0 but steps={steps} is too small.",
            )
        agent_steps = steps - exploration_steps

        if avail_steps < agent_steps:
            self.logger.log(
                f"Requested {agent_steps} transitions but only {avail_steps} in buffer."
                f" Sampling {agent_steps - avail_steps} additional transitions.",
            )
            sample_until = make_sample_until(
                min_timesteps=agent_steps - avail_steps,
                min_episodes=None,
            )
            algo_venv = self.algorithm.get_env()
            assert algo_venv is not None
            generate_trajectories(
                self.algorithm,
                algo_venv,
                sample_until=sample_until,
                deterministic_policy=False,
                rng=self.rng,
            )
            additional_trajs, _ = self.buffering_wrapper_with_reward_wrapper.pop_finished_trajectories()
            agent_trajs = list(agent_trajs) + list(additional_trajs) #4752

        agent_trajs = _get_trajectories(agent_trajs, agent_steps)

        trajectories = list(agent_trajs)
        if exploration_steps > 0:
            self.logger.log(f"Sampling {exploration_steps} exploratory transitions.")
            sample_until = generate_trajectories(
                min_timesteps=exploration_steps,
                min_episodes=None,
            )
            algo_venv = self.algorithm.get_env()
            assert algo_venv is not None
            generate_trajectories(
                policy=self.exploration_wrapper,
                venv=algo_venv,
                sample_until=sample_until,
                deterministic_policy=False,
                rng=self.rng,
            )
            exploration_trajs, _ = self.buffering_wrapper_with_reward_wrapper.pop_finished_trajectories()
            exploration_trajs = _get_trajectories(exploration_trajs, exploration_steps)
            trajectories.extend(list(exploration_trajs))

        return trajectories
    
    @property
    def logger(self) -> imit_logger.HierarchicalLogger:
        return super().logger

    @logger.setter
    def logger(self, value: imit_logger.HierarchicalLogger) -> None:
        self._logger = value
        self.algorithm.set_logger(self.logger)


def _get_trajectories(
    trajectories: Sequence[RecurrentTrajectoryWithRew],
    steps: int,
) -> Sequence[RecurrentTrajectoryWithRew]:
    if steps == 0:
        return []

    available_steps = sum(len(traj) for traj in trajectories)
    if available_steps < steps:
        raise RuntimeError(
            f"Asked for {steps} transitions but only {available_steps} available",
        )
    steps_cumsum = np.cumsum([len(traj) for traj in trajectories])
    idx = int((steps_cumsum >= steps).argmax())
    trajectories = trajectories[: idx + 1]
    assert sum(len(traj) for traj in trajectories) >= steps
    return trajectories

class RecurrentPreferenceModel(nn.Module):

    def __init__(
        self,
        model: reward_nets.RewardNet,
        noise_prob: float = 0.0,
        discount_factor: float = 1.0,
        threshold: float = 50,
        allow_variable_horizon : bool = False,
        member_indx: int = 0 ,
    ) -> None:
        super().__init__()
        self.model = model
        self.noise_prob = noise_prob
        self.discount_factor = discount_factor
        self.threshold = threshold
        base_model = get_base_model(model)
        self.ensemble_model = None
        if isinstance(base_model, recurrent_reward_nets.RecurrentRewardEnsemble):
            is_base = model is base_model
            is_std_wrapper = (
                isinstance(model, reward_nets.AddSTDRewardWrapper)
                and model.base is base_model
            )

            if not (is_base or is_std_wrapper):
                raise ValueError(
                    "RewardEnsemble can only be wrapped"
                    f" by AddSTDRewardWrapper but found {type(model).__name__}.",
                )
            self.ensemble_model = base_model
            self.member_pref_models = []
            for indx ,member in enumerate(self.ensemble_model.members):
                member_pref_model = RecurrentPreferenceModel(
                    cast(reward_nets.RewardNet, member),  # nn.ModuleList is not generic
                    self.noise_prob,
                    self.discount_factor,
                    self.threshold,
                    member_indx = indx
                )
                self.member_pref_models.append(member_pref_model)
        self.allow_variable_horizon = allow_variable_horizon
        self.member_indx = member_indx

    def forward(
        self,
        fragment_pairs: Sequence[RecurrentTrajectoryPair],
    ) -> Tuple[th.Tensor, Optional[th.Tensor]]:
        probs = th.empty(len(fragment_pairs), dtype=th.float32)
        gt_reward_available = _trajectory_pair_includes_reward(fragment_pairs[0])
        if gt_reward_available:
            gt_probs = th.empty(len(fragment_pairs), dtype=th.float32)

        for i, fragment in enumerate(fragment_pairs):
            frag1, frag2 = fragment
            trans1 = flatten_trajectories([frag1])
            trans2 = flatten_trajectories([frag2])
            rews1 = self.rewards(trans1) # predict rewards by reward model
            rews2 = self.rewards(trans2)
            probs[i] = self.probability(rews1, rews2)
            if gt_reward_available:
                frag1 = cast(RecurrentTrajectoryWithRew, frag1)
                frag2 = cast(RecurrentTrajectoryWithRew, frag2)
                gt_rews_1 = th.from_numpy(frag1.rews)
                gt_rews_2 = th.from_numpy(frag2.rews)
                gt_probs[i] = self.probability(gt_rews_1, gt_rews_2)

        return probs, (gt_probs if gt_reward_available else None)
    
    def rewards(self, transitions: RecurrentTransitions) -> th.Tensor:

        state = assert_not_dictobs(transitions.obs)
        action = transitions.acts
        next_state = assert_not_dictobs(transitions.next_obs)
        done = transitions.dones
        hidden_state = transitions.hidden_states
        if self.ensemble_model is not None:
            rews_np, _ = self.ensemble_model.predict_processed_all(
                state,
                action,
                next_state,
                done,
                hidden_state
            )
            assert rews_np.shape == (len(state), self.ensemble_model.num_members)
            rews = util.safe_to_tensor(rews_np).to(self.ensemble_model.device)

        else:
            if len(hidden_state.shape)>3:
                hidden_state = hidden_state[self.member_indx].swapaxes(0,1)
            preprocessed = self.model.preprocess(state, action, next_state, done, hidden_state)
            rews, _ = self.model(*preprocessed)
            assert rews.shape == (len(state),)
        return rews
    
    def probability(self, rews1: th.Tensor, rews2: th.Tensor) -> th.Tensor:

        expected_dims = 2 if self.ensemble_model is not None else 1
        assert rews1.ndim == rews2.ndim == expected_dims
        if self.allow_variable_horizon:
            which_min = min(len(rews2),len(rews1))
            rews2 = rews2[:which_min]
            rews1 = rews1[:which_min]
        if self.discount_factor == 1:
            returns_diff = (rews2 - rews1).sum(axis=0)  
        else:
            device = rews1.device
            assert device == rews2.device
            discounts = self.discount_factor ** th.arange(len(rews1), device=device)
            if self.ensemble_model is not None:
                discounts = discounts.reshape(-1, 1)
            returns_diff = (discounts * (rews2 - rews1)).sum(axis=0)
        returns_diff = th.clip(returns_diff, -self.threshold, self.threshold)
        model_probability = 1 / (1 + returns_diff.exp())
        probability = self.noise_prob * 0.5 + (1 - self.noise_prob) * model_probability
        if self.ensemble_model is not None:
            assert probability.shape == (self.model.num_members,)
        else:
            assert probability.shape == ()
        return probability


def _trajectory_pair_includes_reward(fragment_pair: RecurrentTrajectoryPair) -> bool:
    """Return true if and only if both fragments in the pair include rewards."""
    frag1, frag2 = fragment_pair
    return isinstance(frag1, RecurrentTrajectoryWithRew) and isinstance(frag2, RecurrentTrajectoryWithRew)

class RecurrentRandomFragmenter(preference_comparisons.Fragmenter):

    def __init__(
        self,
        rng: np.random.Generator,
        warning_threshold: int = 10,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        allow_variable_horizon: bool =  False,
    ) -> None:
        self.allow_variable_horizon = allow_variable_horizon
        super().__init__(custom_logger)
        self.rng = rng
        self.warning_threshold = warning_threshold

    def __call__(
        self,
        trajectories: Sequence[RecurrentTrajectoryWithRew],
        fragment_length: int,
        num_pairs: int,
    ) -> Sequence[RecurrentTrajectoryWithRewPair]:
        fragments: List[RecurrentTrajectoryWithRew] = []
        if self.allow_variable_horizon:
            trajectories = [traj for traj in trajectories]
        else:           
            prev_num_trajectories = len(trajectories)
            trajectories = [traj for traj in trajectories if len(traj) >= fragment_length]
            
            if len(trajectories) == 0:
                raise ValueError(
                    "No trajectories are long enough for the desired fragment length "
                    f"of {fragment_length}.",
                )
            num_discarded = prev_num_trajectories - len(trajectories)
            if num_discarded:
                self.logger.log(
                    f"Discarded {num_discarded} out of {prev_num_trajectories} "
                    "trajectories because they are shorter than the desired length "
                    f"of {fragment_length}.",
                )

        weights = [len(traj) for traj in trajectories]

        num_transitions = 2 * num_pairs * fragment_length
        if sum(weights) < num_transitions:
            self.logger.warn(
                "Fewer transitions available than needed for desired number "
                "of fragment pairs. Some transitions will appear multiple times."
                f"sum(weights):{sum(weights)}, num_transitions: {num_transitions}",
            )
        elif (
            self.warning_threshold
            and sum(weights) < self.warning_threshold * num_transitions
        ):
            self.logger.warn(
                f"Samples will contain {num_transitions} transitions in total "
                f"and only {sum(weights)} are available. "
                f"Because we sample with replacement, a significant number "
                "of transitions are likely to appear multiple times.",
            )

        for _ in range(2 * num_pairs):
            traj = self.rng.choice(
                trajectories,  # type: ignore[arg-type]
                p=np.array(weights) / sum(weights),
            )
            n = len(traj)
            if self.allow_variable_horizon:
                if n>fragment_length:
                    start = self.rng.integers(0, n - fragment_length, endpoint=True)
                    end = start + fragment_length
                else:
                    start = 0
                    end = n
                
            else:
                start = self.rng.integers(0, n - fragment_length, endpoint=True)
                end = start + fragment_length
            terminal = (end == n) and traj.terminal
            fragment = RecurrentTrajectoryWithRew(
            obs=traj.obs[start : end + 1],
            acts=traj.acts[start:end],
            infos=traj.infos[start:end] if traj.infos is not None else None,
            rews=traj.rews[start:end],
            hidden_states=traj.hidden_states[start:end].swapaxes(0, 1), ## may be checking here for ensembeling
            terminal=terminal,
            )

            fragments.append(fragment)
        iterator = iter(fragments)
        return list(zip(iterator, iterator))


class RecurrentActiveSelectionFragmenter(preference_comparisons.Fragmenter):
    def __init__(
        self,
        preference_model: RecurrentPreferenceModel,
        base_fragmenter: preference_comparisons.Fragmenter,
        fragment_sample_factor: float,
        uncertainty_on: str = "logit",
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ) -> None:

        super().__init__(custom_logger=custom_logger)
        if preference_model.ensemble_model is None:
            raise ValueError(
                "RecurrentPreferenceModel not wrapped over an ensemble of networks.",
            )
        self.preference_model = preference_model
        self.base_fragmenter = base_fragmenter
        self.allow_variable_horizon = base_fragmenter.allow_variable_horizon
        self.fragment_sample_factor = fragment_sample_factor
        self._uncertainty_on = uncertainty_on
        if not (uncertainty_on in ["logit", "probability", "label"]):
            self.raise_uncertainty_on_not_supported()

    @property
    def uncertainty_on(self) -> str:
        return self._uncertainty_on

    def raise_uncertainty_on_not_supported(self) -> NoReturn:
        raise ValueError(
            f"""{self.uncertainty_on} not supported.
            `uncertainty_on` should be from `logit`, `probability`, or `label`""",
        )

    def __call__(
        self,
        trajectories: Sequence[RecurrentTrajectoryWithRew],
        fragment_length: int,
        num_pairs: int,
    ) -> Sequence[RecurrentTrajectoryWithRewPair]:
        fragments_to_sample = int(self.fragment_sample_factor * num_pairs)
        fragment_pairs = self.base_fragmenter(
            trajectories=trajectories,
            fragment_length=fragment_length,
            num_pairs=fragments_to_sample,
        )
        var_estimates = np.zeros(len(fragment_pairs))
        for i, fragment in enumerate(fragment_pairs):
            frag1, frag2 = fragment
            trans1 = flatten_trajectories([frag1])
            trans2 = flatten_trajectories([frag2])
            with th.no_grad():
                rews1 = self.preference_model.rewards(trans1)
                rews2 = self.preference_model.rewards(trans2)
            var_estimate = self.variance_estimate(rews1, rews2)
            var_estimates[i] = var_estimate
        fragment_idxs = np.argsort(var_estimates)[::-1]  
        return [fragment_pairs[idx] for idx in fragment_idxs[:num_pairs]]

    def variance_estimate(self, rews1: th.Tensor, rews2: th.Tensor) -> float:
        if self.allow_variable_horizon:
            which_min = min(len(rews1), len(rews2))
            rews1 ,rews2 = rews1[:which_min], rews2[:which_min]  

        if self.uncertainty_on == "logit":
            returns1, returns2 = rews1.sum(0), rews2.sum(0)
            var_estimate = (returns1 - returns2).var().item()
        else:  
            probs = self.preference_model.probability(rews1, rews2)
            probs_np = probs.cpu().numpy()
            assert probs_np.shape == (self.preference_model.model.num_members,)
            if self.uncertainty_on == "probability":
                var_estimate = probs_np.var()
            elif self.uncertainty_on == "label":
                preds = (probs_np > 0.5).astype(np.float32)
                prob_estimate = preds.mean()
                var_estimate = prob_estimate * (1 - prob_estimate)
            else:
                self.raise_uncertainty_on_not_supported()
        return var_estimate

class RecurrentSyntheticGatherer(preference_comparisons.PreferenceGatherer):

    def __init__(
        self,
        temperature: float = 1,
        discount_factor: float = 1,
        sample: bool = True,
        rng: Optional[np.random.Generator] = None,
        threshold: float = 50,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        allow_variable_horizon: bool =  False,
    ) -> None:
        super().__init__(custom_logger=custom_logger)
        self.temperature = temperature
        self.discount_factor = discount_factor
        self.sample = sample
        self.rng = rng
        self.threshold = threshold
        self.allow_variable_horizon = allow_variable_horizon

        if self.sample and self.rng is None:
            raise ValueError("If `sample` is True, then `rng` must be provided.")
        
    def __call__(self, fragment_pairs: Sequence[RecurrentTrajectoryWithRewPair]) -> np.ndarray:
        returns1, returns2 = self._reward_sums(fragment_pairs)
        if self.temperature == 0:
            return (np.sign(returns1 - returns2) + 1) / 2

        returns1 /= self.temperature
        returns2 /= self.temperature

        returns_diff = np.clip(returns2 - returns1, -self.threshold, self.threshold)
        model_probs = 1 / (1 + np.exp(returns_diff))
        entropy = -(
            special.xlogy(model_probs, model_probs)
            + special.xlogy(1 - model_probs, 1 - model_probs)
        ).mean()
        self.logger.record("entropy", entropy)

        if self.sample:
            assert self.rng is not None
            return self.rng.binomial(n=1, p=model_probs).astype(np.float32)
        return model_probs

    def _reward_sums(self, fragment_pairs) -> Tuple[np.ndarray, np.ndarray]:
        rews1 = [] 
        rews2 = [] 
        for f1, f2 in fragment_pairs:
            if self.allow_variable_horizon:
                which_min = min(len(f1.rews),len(f2.rews))
                rew_1 =np.array(f1.rews)[:which_min]
                rew_2 = np.array(f2.rews)[:which_min]
            else:
                rew_1 = f1.rews
                rew_2 = f2.rews
            rews1.append(discounted_sum(rew_1, self.discount_factor))
            rews2.append(discounted_sum(rew_2, self.discount_factor))
        return np.array(rews1, dtype=np.float32), np.array(rews2, dtype=np.float32)

class RecurrentPreferenceDataset(preference_comparisons.PreferenceDataset):
    def __init__(self, max_size: Optional[int] = None) -> None:
        self.fragments1: List[RecurrentTrajectoryWithRew] = []
        self.fragments2: List[RecurrentTrajectoryWithRew] = []
        self.max_size = max_size
        self.preferences: np.ndarray = np.array([])

    def push(
        self,
        fragments: Sequence[RecurrentTrajectoryWithRewPair],
        preferences: np.ndarray,
    ) -> None:
        fragments1, fragments2 = zip(*fragments)
        if preferences.shape != (len(fragments),):
            raise ValueError(
                f"Unexpected preferences shape {preferences.shape}, "
                f"expected {(len(fragments),)}",
            )
        if preferences.dtype != np.float32:
            raise ValueError("preferences should have dtype float32")

        self.fragments1.extend(fragments1)
        self.fragments2.extend(fragments2)
        self.preferences = np.concatenate((self.preferences, preferences))

        # Evict old samples if the dataset is at max capacity
        if self.max_size is not None:
            extra = len(self.preferences) - self.max_size
            if extra > 0:
                self.fragments1 = self.fragments1[extra:]
                self.fragments2 = self.fragments2[extra:]
                self.preferences = self.preferences[extra:]

    @overload
    def __getitem__(self, key: int) -> Tuple[RecurrentTrajectoryWithRewPair, float]:
        pass

    @overload
    def __getitem__(
        self,
        key: slice,
    ) -> Tuple[Pair[Sequence[RecurrentTrajectoryWithRew]], Sequence[float]]:
        pass

    def __getitem__(self, key):
        return (self.fragments1[key], self.fragments2[key]), self.preferences[key]

    def __len__(self) -> int:
        assert len(self.fragments1) == len(self.fragments2) == len(self.preferences)
        return len(self.fragments1)

    def save(self, path: AnyPath) -> None:
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path: AnyPath) -> "RecurrentPreferenceDataset":
        with open(path, "rb") as file:
            return pickle.load(file)

def preference_collate_fn(
    batch: Sequence[Tuple[RecurrentTrajectoryWithRewPair, float]],
) -> Tuple[Sequence[RecurrentTrajectoryWithRewPair], np.ndarray]:
    fragment_pairs, preferences = zip(*batch)
    return list(fragment_pairs), np.array(preferences)

class RecurrentCrossEntropyRewardLoss(preference_comparisons.CrossEntropyRewardLoss):
    def __init__(self)  -> None :
        super().__init__()

    def forward(
        self,
        fragment_pairs: Sequence[RecurrentTrajectoryPair],
        preferences: np.ndarray,
        preference_model: preference_comparisons.PreferenceModel,
    ) -> preference_comparisons.LossAndMetrics:
        probs, gt_probs = preference_model(fragment_pairs)
        predictions = probs > 0.5
        preferences_th = th.as_tensor(preferences, dtype=th.float32)
        ground_truth = preferences_th > 0.5
        metrics = {}
        metrics["accuracy"] = (predictions == ground_truth).float().mean()
        if gt_probs is not None:
            metrics["gt_reward_loss"] = th.nn.functional.binary_cross_entropy(
                gt_probs,
                preferences_th,
            )
        metrics = {key: value.detach().cpu() for key, value in metrics.items()}
        return preference_comparisons.LossAndMetrics(
            loss=th.nn.functional.binary_cross_entropy(probs, preferences_th),
            metrics=metrics,
        )

    
class RecurrentBasicRewardTrainer(preference_comparisons.RewardTrainer):
    regularizer: Optional[regularizers.Regularizer]

    def __init__(
        self,
        preference_model: RecurrentPreferenceModel,
        loss: preference_comparisons.RewardLoss,
        rng: np.random.Generator,
        batch_size: int = 32,
        minibatch_size: Optional[int] = None,
        epochs: int = 1,
        lr: float = 1e-3,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        regularizer_factory: Optional[regularizers.RegularizerFactory] = None,
    ) -> None:
        super().__init__(preference_model, custom_logger)
        self.loss = loss
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size or batch_size
        if self.batch_size % self.minibatch_size != 0:
            raise ValueError("Batch size must be a multiple of minibatch size.")
        self.epochs = epochs
        self.optim = th.optim.AdamW(self._preference_model.parameters(), lr=lr)
        self.rng = rng
        self.regularizer = (
            regularizer_factory(optimizer=self.optim, logger=self.logger)
            if regularizer_factory is not None
            else None
        )

    def _make_data_loader(self, dataset: data_th.Dataset) -> data_th.DataLoader:
        """Make a dataloader."""
        return data_th.DataLoader(
            dataset,
            batch_size=self.minibatch_size,
            shuffle=True,
            collate_fn=preference_collate_fn,
        )

    @property
    def requires_regularizer_update(self) -> bool:
        """Whether the regularizer requires updating.

        Returns:
            If true, this means that a validation dataset will be used.
        """
        return self.regularizer is not None and self.regularizer.val_split is not None

    def _train(
        self,
        dataset: preference_comparisons.PreferenceDataset,
        epoch_multiplier: float = 1.0,
    ) -> None:
        """Trains for `epoch_multiplier * self.epochs` epochs over `dataset`."""
        if self.regularizer is not None and self.regularizer.val_split is not None:
            val_length = int(len(dataset) * self.regularizer.val_split)
            train_length = len(dataset) - val_length
            if val_length < 1 or train_length < 1:
                raise ValueError(
                    "Not enough data samples to split into training and validation, "
                    "or the validation split is too large/small. "
                    "Make sure you've generated enough initial preference data. "
                    "You can adjust this through initial_comparison_frac in "
                    "PreferenceComparisons.",
                )
            train_dataset, val_dataset = data_th.random_split(
                dataset,
                lengths=[train_length, val_length],
                # we convert the numpy generator to the pytorch generator.
                generator=th.Generator().manual_seed(util.make_seeds(self.rng)),
            )
            dataloader = self._make_data_loader(train_dataset)
            val_dataloader = self._make_data_loader(val_dataset)
        else:
            dataloader = self._make_data_loader(dataset)
            val_dataloader = None

        epochs = round(self.epochs * epoch_multiplier)

        assert epochs > 0, "Must train for at least one epoch."
        with self.logger.accumulate_means("reward"):
            for epoch_num in tqdm(range(epochs), desc="Training reward model"):
                with self.logger.add_key_prefix(f"epoch-{epoch_num}"):
                    train_loss = 0.0
                    accumulated_size = 0
                    self.optim.zero_grad()
                    for fragment_pairs, preferences in dataloader:
                        with self.logger.add_key_prefix("train"):
                            loss = self._training_inner_loop(
                                fragment_pairs,
                                preferences,
                            )
                            loss *= len(fragment_pairs) / self.batch_size

                        train_loss += loss.item()
                        if self.regularizer:
                            self.regularizer.regularize_and_backward(loss)
                        else:
                            loss.backward()

                        accumulated_size += len(fragment_pairs)
                        if accumulated_size >= self.batch_size:
                            self.optim.step()
                            self.optim.zero_grad()
                            accumulated_size = 0
                    if accumulated_size != 0:
                        self.optim.step()  # if there remains an incomplete batch

                    if not self.requires_regularizer_update:
                        continue
                    assert val_dataloader is not None
                    assert self.regularizer is not None

                    val_loss = 0.0
                    for fragment_pairs, preferences in val_dataloader:
                        with self.logger.add_key_prefix("val"):
                            val_loss += self._training_inner_loop(
                                fragment_pairs,
                                preferences,
                            ).item()
                    self.regularizer.update_params(train_loss, val_loss)

        keys = list(self.logger.name_to_value.keys())
        outer_prefix = self.logger.get_accumulate_prefixes()
        for key in keys:
            base_path = f"{outer_prefix}reward/"  # existing prefix + accum_means ctx
            epoch_path = f"mean/{base_path}epoch-{epoch_num}/"  # mean for last epoch
            final_path = f"{base_path}final/"  # path to record last epoch
            pattern = rf"{epoch_path}(.+)"
            if regex_match := re.match(pattern, key):
                (key_name,) = regex_match.groups()
                val = self.logger.name_to_value[key]
                new_key = f"{final_path}{key_name}"
                self.logger.record(new_key, val)

    def _training_inner_loop(
        self,
        fragment_pairs: Sequence[RecurrentTrajectoryPair],
        preferences: np.ndarray,
    ) -> th.Tensor:
        output = self.loss.forward(fragment_pairs, preferences, self._preference_model)
        loss = output.loss
        self.logger.record("loss", loss.item())
        for name, value in output.metrics.items():
            self.logger.record(name, value.item())
        return loss
    

class RecurrentEnsembleTrainer(RecurrentBasicRewardTrainer):

    def __init__(
        self,
        preference_model: RecurrentPreferenceModel,
        loss: preference_comparisons.RewardLoss,
        rng: np.random.Generator,
        batch_size: int = 32,
        minibatch_size: Optional[int] = None,
        epochs: int = 1,
        lr: float = 1e-3,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        regularizer_factory: Optional[regularizers.RegularizerFactory] = None,
    ) -> None:
        if preference_model.ensemble_model is None:
            raise TypeError(
                "RecurrentPreferenceModel of a RewardEnsemble expected by EnsembleTrainer.",
            )

        super().__init__(
            preference_model,
            loss=loss,
            batch_size=batch_size,
            minibatch_size=minibatch_size,
            epochs=epochs,
            lr=lr,
            custom_logger=custom_logger,
            rng=rng,
            regularizer_factory=regularizer_factory,
        )
        self.member_trainers = []
        self._call_iter = 0
        for member_pref_model in self._preference_model.member_pref_models:
            reward_trainer = RecurrentBasicRewardTrainer(
                member_pref_model,
                loss=loss,
                batch_size=batch_size,
                minibatch_size=minibatch_size,
                epochs=epochs,
                lr=lr,
                custom_logger=self.logger,
                regularizer_factory=regularizer_factory,
                rng=self.rng,
            )
            self.member_trainers.append(reward_trainer)

    @property
    def logger(self) -> imit_logger.HierarchicalLogger:
        return super().logger

    @logger.setter
    def logger(self, custom_logger: imit_logger.HierarchicalLogger) -> None:
        self._logger = custom_logger
        for member_trainer in self.member_trainers:
            member_trainer.logger = custom_logger

    def _train(self, dataset: preference_comparisons.PreferenceDataset, epoch_multiplier: float = 1.0) -> None:
        """Trains for `epoch_multiplier * self.epochs` epochs over `dataset`."""
        sampler = data_th.RandomSampler(
            dataset,
            replacement=True,
            num_samples=len(dataset),
            # we convert the numpy generator to the pytorch generator.
            generator=th.Generator().manual_seed(util.make_seeds(self.rng)),
        )
        for member_idx in range(len(self.member_trainers)):
            # sampler gives new indexes on every call
            bagging_dataset = data_th.Subset(dataset, list(sampler))
            with self.logger.add_accumulate_prefix(f"member-{member_idx}"):
                self.member_trainers[member_idx].train(
                                                        bagging_dataset,
                                                        epoch_multiplier=epoch_multiplier,
                                                    )
            
        
        self._call_iter += 1
        metrics = defaultdict(list)
        keys = list(self.logger.name_to_value.keys())
        for key in keys:
            if re.match(r"member-(\d+)/reward/(.+)", key) and "final" in key:
                val = self.logger.name_to_value[key]
                key_list = key.split("/")
                key_list.pop(0)
                metrics["/".join(key_list)].append(val)

        for k, v in metrics.items():
            self.logger.record(k, np.mean(v))
            self.logger.record(k + "_std", np.std(v))

def get_base_model(reward_model: reward_nets.RewardNet) -> reward_nets.RewardNet:
    base_model = reward_model
    while hasattr(base_model, "base"):
        base_model = cast(reward_nets.RewardNet, base_model.base)
    return base_model

def _make_reward_trainer(
    preference_model: RecurrentPreferenceModel,
    loss: preference_comparisons.RewardLoss,
    rng: np.random.Generator,
    reward_trainer_kwargs: Optional[Mapping[str, Any]] = None,
) -> preference_comparisons.RewardTrainer:
    """Construct the correct type of reward trainer for this reward function."""
    if reward_trainer_kwargs is None:
        reward_trainer_kwargs = {}

    if preference_model.ensemble_model is not None:
        return RecurrentEnsembleTrainer(
            preference_model,
            loss,
            rng=rng,
            **reward_trainer_kwargs,
        )
    else:
        return RecurrentBasicRewardTrainer(
            preference_model,
            loss=loss,
            rng=rng,
            **reward_trainer_kwargs,
        )


QUERY_SCHEDULES = preference_comparisons.QUERY_SCHEDULES
from imitation.algorithms import base

class RecurrentPreferenceComparisons(base.BaseImitationAlgorithm):

    def __init__(
        self,
        trajectory_generator: preference_comparisons.TrajectoryGenerator,
        reward_model: reward_nets.RewardNet,
        num_iterations: int,
        fragmenter: Optional[preference_comparisons.Fragmenter] = None,
        preference_gatherer: Optional[preference_comparisons.PreferenceGatherer] = None,
        reward_trainer: Optional[preference_comparisons.RewardTrainer] = None,
        comparison_queue_size: Optional[int] = None,
        fragment_length: int = 100,
        transition_oversampling: float = 1,
        initial_comparison_frac: float = 0.1,
        initial_epoch_multiplier: float = 200.0,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        allow_variable_horizon: bool = False,
        rng: Optional[np.random.Generator] = None,
        query_schedule: Union[str, type_aliases.Schedule] = "hyperbolic",
        tensorboard = None
    ) -> None:
        
        super().__init__(
            custom_logger=custom_logger,
            allow_variable_horizon=allow_variable_horizon,
        )
        self._iteration = 0

        self.model = reward_model
        self.rng = rng
        self.tensorboard = tensorboard
        has_any_rng_args_none = None in (
            preference_gatherer,
            fragmenter,
            reward_trainer,
        )

        if self.rng is None and has_any_rng_args_none:
            raise ValueError(
                "If you don't provide a random state, you must provide your own "
                "seeded fragmenter, preference gatherer, and reward_trainer. "
                "You can initialize a random state with `np.random.default_rng(seed)`.",
            )
        elif self.rng is not None and not has_any_rng_args_none:
            raise ValueError(
                "If you provide your own fragmenter, preference gatherer, "
                "and reward trainer, you don't need to provide a random state.",
            )
        if reward_trainer is None:
            assert self.rng is not None
            preference_model = RecurrentPreferenceModel(reward_model)
            loss = RecurrentCrossEntropyRewardLoss()
            self.reward_trainer = _make_reward_trainer(
                preference_model,
                loss,
                rng=self.rng,
            )
        else:
            self.reward_trainer = reward_trainer

        self.reward_trainer.logger = self.logger
        self.trajectory_generator = trajectory_generator
        self.trajectory_generator.logger = self.logger
        if fragmenter:
            self.fragmenter = fragmenter
        else:
            assert self.rng is not None
            self.fragmenter = RecurrentRandomFragmenter(
                custom_logger=self.logger,
                rng=self.rng,
            )
        self.fragmenter.logger = self.logger
        if preference_gatherer:
            self.preference_gatherer = preference_gatherer
        else:
            assert self.rng is not None
            self.preference_gatherer = RecurrentSyntheticGatherer(
                custom_logger=self.logger,
                rng=self.rng,
            )

        self.preference_gatherer.logger = self.logger

        self.fragment_length = fragment_length
        self.initial_comparison_frac = initial_comparison_frac
        self.initial_epoch_multiplier = initial_epoch_multiplier
        self.num_iterations = num_iterations
        self.transition_oversampling = transition_oversampling
        if callable(query_schedule):
            self.query_schedule = query_schedule
        elif query_schedule in QUERY_SCHEDULES:
            self.query_schedule = QUERY_SCHEDULES[query_schedule]
        else:
            raise ValueError(f"Unknown query schedule: {query_schedule}")

        self.dataset = RecurrentPreferenceDataset(max_size=comparison_queue_size)


    def train(
        self,
        total_timesteps: int,
        total_comparisons: int,
        callback: MaybeCallback = None,
    ) -> Mapping[str, Any]:
        initial_comparisons = int(total_comparisons * self.initial_comparison_frac)
        total_comparisons -= initial_comparisons
        vec_schedule = np.vectorize(self.query_schedule)
        unnormalized_probs = vec_schedule(np.linspace(0, 1, self.num_iterations))
        probs = unnormalized_probs / np.sum(unnormalized_probs)
        shares = util.oric(probs * total_comparisons)
        schedule = [initial_comparisons] + shares.tolist()
        print(f"Query schedule: {schedule}")

        timesteps_per_iteration, extra_timesteps = divmod(
            total_timesteps,
            self.num_iterations,
        )
        reward_loss = None
        reward_accuracy = None

        for i, num_pairs in enumerate(schedule):
            num_steps = math.ceil(
                self.transition_oversampling * 2 * num_pairs * self.fragment_length,
            )
            self.logger.log(
                f"Collecting {2 * num_pairs} fragments ({num_steps} transitions)",
            )
            trajectories = self.trajectory_generator.sample(num_steps)
            horizons = (len(traj) for traj in trajectories if traj.terminal)
            self._check_fixed_horizon(horizons)
            self.logger.log("Creating fragment pairs")
            fragments = self.fragmenter(trajectories, self.fragment_length, num_pairs)

            with self.logger.accumulate_means("preferences"):
                self.logger.log("Gathering preferences")
                preferences = self.preference_gatherer(fragments) ## TODO it can exchange self.preference_gatherer
            self.dataset.push(fragments, preferences)
            self.logger.log(f"Dataset now contains {len(self.dataset)} comparisons")

            epoch_multiplier = 1.0
            if i == 0:
                epoch_multiplier = self.initial_epoch_multiplier
            self.reward_trainer.train(self.dataset, epoch_multiplier=epoch_multiplier)
            base_key = self.logger.get_accumulate_prefixes() + "reward/final/train"
            assert f"{base_key}/loss" in self.logger.name_to_value
            assert f"{base_key}/accuracy" in self.logger.name_to_value
            reward_loss = self.logger.name_to_value[f"{base_key}/loss"]
            reward_accuracy = self.logger.name_to_value[f"{base_key}/accuracy"]
            if self.tensorboard is not None: 
                self.tensorboard.add_scalar('gru/loss_final',reward_loss, i)
                self.tensorboard.add_scalar('gru/accuracy_final',reward_accuracy, i)
            num_steps = timesteps_per_iteration
            if i == self.num_iterations - 1:
                num_steps += extra_timesteps
            with self.logger.accumulate_means("agent"):
                self.logger.log(f"Training agent for {num_steps} timesteps")
                self.trajectory_generator.train(steps=num_steps)

            self.logger.dump(self._iteration)
            if callback:
                callback(self._iteration)
            self._iteration += 1

        return {"reward_loss": reward_loss, "reward_accuracy": reward_accuracy}

class DictRecurrentPreferenceModel(RecurrentPreferenceModel):
    
    def rewards(self, transitions: RecurrentTransitions) -> th.Tensor:

        state = transitions.obs
        action = transitions.acts
        next_state = transitions.next_obs
        done = transitions.dones
        hidden_state = transitions.hidden_states
        if self.ensemble_model is not None:
            rews_np, _ = self.ensemble_model.predict_processed_all(
                state,
                action,
                next_state,
                done,
                hidden_state
            )
            assert rews_np.shape == (len(state), self.ensemble_model.num_members)
            rews = util.safe_to_tensor(rews_np).to(self.ensemble_model.device)

        else:
            if len(hidden_state.shape)>3:
                hidden_state = hidden_state[self.member_indx].swapaxes(0,1)
            preprocessed = self.model.preprocess(state, action, next_state, done, hidden_state)
            rews, _ = self.model(*preprocessed)
            for key, state_ in state.items():
                if isinstance(state, dict):
                    assert rews.shape == (len(state_[key]),)
        return rews
