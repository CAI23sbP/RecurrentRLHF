from typing import (
    TypeVar,
    cast,

)
T = TypeVar("T")
from imitation.algorithms.preference_comparisons import PreferenceModel, get_base_model
from imitation.rewards import reward_nets 
import torch.nn as nn 

class DictPreferenceModel(nn.Module):
    """Class to convert two fragments' rewards into preference probability."""

    def __init__(
        self,
        model: reward_nets.RewardNet,
        noise_prob: float = 0.0,
        discount_factor: float = 1.0,
        threshold: float = 50,
    ) -> None:
        super().__init__()
        self.model = model
        self.noise_prob = noise_prob
        self.discount_factor = discount_factor
        self.threshold = threshold
        base_model = get_base_model(model)
        self.ensemble_model = None

        if isinstance(base_model, reward_nets.RewardEnsemble):
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
            for member in self.ensemble_model.members:
                member_pref_model = PreferenceModel(
                    cast(reward_nets.RewardNet, member),  # nn.ModuleList is not generic
                    self.noise_prob,
                    self.discount_factor,
                    self.threshold,
                )
                self.member_pref_models.append(member_pref_model)
