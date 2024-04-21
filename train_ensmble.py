from imitation.algorithms import preference_comparisons
from test_net.non_recurrent_net import CustomBasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from imitation.policies.base import NormalizeFeaturesExtractor
from stable_baselines3 import PPO
import numpy as np
import gymnasium as gym 
import recurrent_preference
from test_net import recurrent_net
import custom_envs
import torch.nn as nn 
import seals 

N_ENVS = 8
N_STEPS = int(2048  / N_ENVS)
BATCH_SIZE = 64
TEST_EPI = 1 
RENDER = True
CLIP_SIZE = 60 
"""
ENV_NAME:
    BipedalWalker family: 
        FHBipedalWalker-v1: absorb observation and reward 
        FHBipedalWalker-v2: only has a truncated (i.e. hasn't a termination) and hasn't absorb series

    CartPole
        seals/CartPole-v0: reference https://github.com/HumanCompatibleAI/seals/blob/master/src/seals/classic_control.py
    
    Pendulum
        Pendulum-v1: reference https://imitation.readthedocs.io/en/latest/tutorials/5_train_preference_comparisons.html
    
    Continuous_MountainCarEnv
        FHContinuous_MountainCarEnv-v0: reference https://github.com/HumanCompatibleAI/seals/blob/master/src/seals/classic_control.py

"""

ENV_NAME = 'FHBipedalWalker-v2' 
HARDCORE = True # it only works BipedalWalker family

max_episode_steps = 500
rng = np.random.default_rng(0)

policy_kwargs = dict(
        net_arch = dict(pi = [32,32], vf = [32,32]),
        activation_fn = nn.ReLU,
        features_extractor_class=NormalizeFeaturesExtractor,
        features_extractor_kwargs=dict(normalize_class=RunningNorm),
        )

venv = make_vec_env(ENV_NAME, rng=rng, 
                    n_envs= N_ENVS, 
                    max_episode_steps = max_episode_steps, 
                    env_make_kwargs = {'hardcore':HARDCORE} if 'BipedalWalker' in ENV_NAME else None
                    )

reward_net = recurrent_net.CustomRewardNet(
    venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
)

fragmenter = recurrent_preference.RecurrentRandomFragmenter(
    warning_threshold=0,
    rng=rng,
)
gatherer = recurrent_preference.RecurrentSyntheticGatherer(rng=rng)
preference_model = recurrent_preference.RecurrentPreferenceModel(reward_net)
reward_trainer = recurrent_preference.RecurrentBasicRewardTrainer(
    preference_model=preference_model,
    loss=recurrent_preference.RecurrentCrossEntropyRewardLoss(),
    epochs=3,
    rng=rng,
)
agent = PPO(
    policy="MlpPolicy",
    env=venv,
    seed=0,
    policy_kwargs = policy_kwargs,
    learning_rate= 0.0004,
    use_sde = False,
    n_steps=N_STEPS,
    batch_size=BATCH_SIZE,
    ent_coef=0.01,
    clip_range=0.1,
    gae_lambda=0.95,
    gamma=0.99,
    n_epochs=10,
)

trajectory_generator = recurrent_preference.RecurrentAgentTrainer(
    algorithm=agent,
    reward_fn=reward_net,
    venv=venv,
    rng=rng,
)

pref_comparisons = recurrent_preference.RecurrentPreferenceComparisons(
    trajectory_generator,
    reward_net,
    num_iterations=60,  
    fragmenter=fragmenter,
    preference_gatherer=gatherer,
    reward_trainer=reward_trainer,
    fragment_length=CLIP_SIZE,
    transition_oversampling=1,
    initial_comparison_frac=0.1,
    allow_variable_horizon=False,
    initial_epoch_multiplier=4,
    query_schedule="hyperbolic",
)
pref_comparisons.train(
    total_timesteps=5_000,
    total_comparisons=200,
)

del pref_comparisons, trajectory_generator, rng, reward_trainer , preference_model, gatherer, fragmenter, venv, reward_net

rng = np.random.default_rng(0)

venv = make_vec_env(ENV_NAME, rng=rng, 
                    n_envs= N_ENVS, 
                    max_episode_steps = max_episode_steps, 
                    env_make_kwargs = {'hardcore':HARDCORE} if 'BipedalWalker' in ENV_NAME else None
                    )

reward_net = CustomBasicRewardNet(
    venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
)

fragmenter = preference_comparisons.RandomFragmenter(
    warning_threshold=0,
    rng=rng,
)
gatherer = preference_comparisons.SyntheticGatherer(rng=rng)
preference_model = preference_comparisons.PreferenceModel(reward_net)
reward_trainer = preference_comparisons.BasicRewardTrainer(
    preference_model=preference_model,
    loss=preference_comparisons.CrossEntropyRewardLoss(),
    epochs=3,
    rng=rng,
)
agent2 = PPO(
    policy="MlpPolicy",
    env=venv,
    seed=0,
    policy_kwargs = policy_kwargs,
    learning_rate= 0.0004,
    use_sde = False,
    n_steps=N_STEPS,
    batch_size=BATCH_SIZE,
    ent_coef=0.01,
    clip_range=0.1,
    gae_lambda=0.95,
    gamma=0.99,
    n_epochs=10,
)

trajectory_generator = preference_comparisons.AgentTrainer(
    algorithm=agent2,
    reward_fn=reward_net,
    venv=venv,
    exploration_frac=0.00,
    rng=rng,
)

pref_comparisons = preference_comparisons.PreferenceComparisons(
    trajectory_generator,
    reward_net,
    num_iterations=60,  # Set to 60 for better performance
    fragmenter=fragmenter,
    preference_gatherer=gatherer,
    reward_trainer=reward_trainer,
    fragment_length=CLIP_SIZE,
    transition_oversampling=1,
    initial_comparison_frac=0.1,
    allow_variable_horizon=False,
    initial_epoch_multiplier=4,
    query_schedule="hyperbolic",
)
pref_comparisons.train(
    total_timesteps=5_000,
    total_comparisons=200,
)

del pref_comparisons, trajectory_generator, rng, reward_trainer , preference_model, gatherer, fragmenter, venv, reward_net

import gymnasium as gym 


agents_rewards = []
agents_epi_lenghts = []
for index ,model in enumerate([agent, agent2]):
    reward_list = []
    epi_lenght_list = []
    if 'BipedalWalker' in ENV_NAME:
        env = gym.make('BipedalWalker-v3', 
                    max_episode_steps = max_episode_steps, 
                    render_mode = 'human',
                    hardcore = HARDCORE
                    )
    else:
        env = gym.make(ENV_NAME, 
            max_episode_steps = max_episode_steps, 
            render_mode = 'human',
            )


    for epi in range(TEST_EPI):
        is_done = False
        observation, _ = env.reset()
        reward_ = 0
        epi_lenght = 0

        while not is_done:
            action, _= model.policy.predict(observation, deterministic = True) 
            observation, reward, terminated, truncated, _ = env.step(action)
            epi_lenght += 1
            reward_ += reward
            env.render()
            if terminated or truncated:
                is_done = True 
        
        reward_list.append(reward_)
        epi_lenght_list.append(epi_lenght)
    del env
    agents_rewards.append([np.mean(reward_list), np.var(reward_list), np.std(reward_list)])
    agents_epi_lenghts.append([np.mean(epi_lenght_list), np.var(epi_lenght_list), np.std(epi_lenght_list)])

print(f'[GRU]\n\
      [reward mean]: {agents_rewards[0][0]}, [reward var]: {agents_rewards[0][1]}, [reward std]: {agents_rewards[0][2]}\n\
        [episode mean]: {agents_epi_lenghts[0][0]}, [episode var]: {agents_epi_lenghts[0][1]}, [episode std]: {agents_epi_lenghts[0][2]}')

print(f'[No GRU]\n\
      [reward mean]: {agents_rewards[1][0]}, [reward var]: {agents_rewards[1][1]}, [reward std]: {agents_rewards[1][2]}\n\
        [episode mean]: {agents_epi_lenghts[1][0]}, [episode var]: {agents_epi_lenghts[1][1]}, [episode std]: {agents_epi_lenghts[1][2]}')