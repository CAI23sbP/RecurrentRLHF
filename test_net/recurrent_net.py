import gymnasium  as gym 
from common.reward_nets.recurrent_reward_nets import RecurrentRewardNet
import torch as th 
from typing import Dict, Any
from stable_baselines3.common import preprocessing
from imitation.util import networks
import torch.nn as nn 

class CustomRewardNet(RecurrentRewardNet):
    
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        use_state: bool = True,
        use_action: bool = True,
        use_next_state: bool = False,
        use_done: bool = False,
        **kwargs
    ):
        super().__init__(observation_space, action_space)

        combined_size = 0

        self.use_state = use_state
        if self.use_state:
            combined_size += preprocessing.get_flattened_obs_dim(observation_space)

        self.use_action = use_action
        if self.use_action:
            combined_size += preprocessing.get_flattened_obs_dim(action_space)

        self.use_next_state = use_next_state
        if self.use_next_state:
            combined_size += preprocessing.get_flattened_obs_dim(observation_space)

        self.use_done = use_done
        if self.use_done:
            combined_size += 1

        n_gru_layers = 1
        gru_hidden_size = 128 
        self.gru = nn.GRU(combined_size, gru_hidden_size, n_gru_layers)

        full_build_output_kwargs: Dict[str, Any] = {
            "hid_sizes": [32, 32],
            "activation": nn.ReLU,
            "in_size": gru_hidden_size,
            "out_size": 1,
            "squeeze_output": True,
        }
        self.output_mlp = networks.build_mlp(**full_build_output_kwargs)

    def forward(self, state, action, next_state, done, hidden_state):
        inputs = []
        if self.use_state:
            inputs.append(th.flatten(state, 1))
        if self.use_action:
            inputs.append(th.flatten(action, 1))
        if self.use_next_state:
            inputs.append(th.flatten(next_state, 1))
        if self.use_done:
            inputs.append(th.reshape(done, [-1, 1]))

        inputs_concat = th.cat(inputs, dim=1)
        latent , hidden_state = self._process_sequence(inputs_concat, hidden_state, done, self.gru)
        outputs = self.output_mlp(latent)
        assert outputs.shape == state.shape[:1]

        return outputs, hidden_state

    