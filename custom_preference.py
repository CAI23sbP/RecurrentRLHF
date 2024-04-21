from imitation.algorithms import preference_comparisons 
from imitation.rewards import reward_nets 
"""
for making tensorboard
"""
from typing import Optional, Union, Callable, Mapping, Any
from imitation.util import logger as imit_logger
from stable_baselines3.common import type_aliases
import numpy as np 
from imitation.util import networks, util
import math 

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
                self.writer.add_scalar('gru/loss_final', reward_loss, i)
                self.writer.add_scalar('gru/accuracy_final', reward_accuracy, i)
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