from imitation.algorithms import preference_comparisons 
from imitation.rewards import reward_nets 
"""
for making tensorboard
"""
from typing import Optional, Union, Callable, Mapping, Any
from imitation.util import logger as imit_logger
from stable_baselines3.common import type_aliases
import numpy as np 
from imitation.util import util
from imitation.data import rollout
import math 
import torch as th
import numpy as np 
from imitation.data.types import (
    TrajectoryWithRew,
    TrajectoryWithRewPair,
)
from typing import (
    Callable,
    Mapping,
    Optional,
    Sequence,
    List,
    Tuple,
    Union,
)
from scipy import special

class CustomRandomFragmenter(preference_comparisons.Fragmenter):

    def __init__(
        self,
        rng: np.random.Generator,
        warning_threshold: int = 10,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        allow_variable_horizon: bool = False
    ) -> None:
        super().__init__(custom_logger)
        self.rng = rng
        self.warning_threshold = warning_threshold
        self.allow_variable_horizon = allow_variable_horizon 

    def __call__(
        self,
        trajectories: Sequence[TrajectoryWithRew],
        fragment_length: int,
        num_pairs: int,
    ) -> Sequence[TrajectoryWithRewPair]:
        fragments: List[TrajectoryWithRew] = []

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
                "of fragment pairs. Some transitions will appear multiple times.",
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
                trajectories, 
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
            fragment = TrajectoryWithRew(
                obs=traj.obs[start : end + 1],
                acts=traj.acts[start:end],
                infos=traj.infos[start:end] if traj.infos is not None else None,
                rews=traj.rews[start:end],
                terminal=terminal,
            )
            fragments.append(fragment)
        iterator = iter(fragments)
        return list(zip(iterator, iterator))


class CustomSyntheticGatherer(preference_comparisons.SyntheticGatherer):

    def __init__(
        self,
        temperature: float = 1,
        discount_factor: float = 1,
        sample: bool = True,
        rng: Optional[np.random.Generator] = None,
        threshold: float = 50,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        allow_variable_horizon : bool = False
    ) -> None:
        super().__init__(temperature, discount_factor, sample, rng, threshold, custom_logger)
        self.allow_variable_horizon = allow_variable_horizon

    def __call__(self, fragment_pairs: Sequence[TrajectoryWithRewPair]) -> np.ndarray:
        """Computes probability fragment 1 is preferred over fragment 2."""
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
            rews1.append(rollout.discounted_sum(rew_1, self.discount_factor))
            rews2.append(rollout.discounted_sum(rew_2, self.discount_factor))
        return np.array(rews1, dtype=np.float32), np.array(rews2, dtype=np.float32)

class CustomPreferenceModel(preference_comparisons.PreferenceModel):

    def __init__(
        self,
        model: reward_nets.RewardNet,
        noise_prob: float = 0.0,
        discount_factor: float = 1.0,
        threshold: float = 50,
        allow_variable_horizon: bool =  False,
    ) -> None:
        self.allow_variable_horizon = allow_variable_horizon
        super().__init__(model, noise_prob, discount_factor, threshold)
    def rewards(self, transitions) -> th.Tensor:

        state = transitions.obs
        action = transitions.acts
        next_state = transitions.next_obs
        done = transitions.dones
        if self.ensemble_model is not None:
            rews_np = self.ensemble_model.predict_processed_all(
                state,
                action,
                next_state,
                done,
            )
            assert rews_np.shape == (len(state), self.ensemble_model.num_members)
            rews = util.safe_to_tensor(rews_np).to(self.ensemble_model.device)

        else:
            preprocessed = self.model.preprocess(state, action, next_state, done)
            rews = self.model(*preprocessed)
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

class CustomPreferenceComparisons(preference_comparisons.PreferenceComparisons):
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
        tensorboard =  None
    ) -> None:
        self.writer = tensorboard
        super().__init__(trajectory_generator, 
                         reward_model,
                         num_iterations,
                         fragmenter,
                         preference_gatherer,
                         reward_trainer,
                        comparison_queue_size,
                        fragment_length,
                        transition_oversampling,
                        initial_comparison_frac,
                        initial_epoch_multiplier,
                        custom_logger,
                        allow_variable_horizon,
                        rng,
                        query_schedule,
                         )

    def train(
        self,
        total_timesteps: int,
        total_comparisons: int,
        callback: Optional[Callable[[int], None]] = None,
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
                preferences = self.preference_gatherer(fragments)
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
            if self.writer is not None:
                self.writer.add_scalar('none_gru/loss_final', reward_loss, i)
                self.writer.add_scalar('none_gru/accuracy_final', reward_accuracy, i)
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