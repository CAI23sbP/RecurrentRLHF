from typing import (
    TypeVar,
    cast,

)
T = TypeVar("T")
from imitation.algorithms.preference_comparisons import PreferenceModel
from imitation.data.types import Transitions, assert_not_dictobs
from imitation.util import util
import torch as th 
import recurrent_preference 
from common.data.recurrent_types import RecurrentTransitions

class DictPreferenceModel(PreferenceModel):

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

