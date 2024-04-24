from typing import (
    TypeVar,
    cast,

)
T = TypeVar("T")
from imitation.algorithms.preference_comparisons import PreferenceModel, get_base_model
from imitation.rewards import reward_nets 
import torch.nn as nn 
from imitation.data.types import Transitions, assert_not_dictobs
from imitation.util import util
import torch as th 
import recurrent_preference 
from common.data.recurrent_types import RecurrentTransitions

class DictPreferenceModel(PreferenceModel):
    """Class to convert two fragments' rewards into preference probability."""

    def rewards(self, transitions: Transitions) -> th.Tensor:
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
            for key, state_ in state.items():
                if isinstance(state, dict):
                    assert rews.shape == (len(state_[key]),)
        return rews

class DictRecurrentPreferenceModel(recurrent_preference.RecurrentPreferenceModel):
    
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
            for key, state_ in state.items():
                if isinstance(state, dict):
                    assert rews.shape == (len(state_[key]),)
        return rews
    
