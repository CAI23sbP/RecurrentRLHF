from imitation.algorithms import preference_comparisons
from test_net.non_recurrent_net import CustomBasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from imitation.policies.base import NormalizeFeaturesExtractor
from imitation.rewards.reward_nets import RewardEnsemble, AddSTDRewardWrapper, RewardNet
from common.reward_nets.recurrent_reward_nets import RecurrentRewardEnsemble, RecurrentAddSTDRewardWrapper
from custom_preference import CustomPreferenceComparisons, CustomPreferenceModel, CustomRandomFragmenter, CustomSyntheticGatherer
from stable_baselines3 import PPO
import numpy as np
import gymnasium as gym 
import recurrent_preference
from test_net import recurrent_net
import custom_envs
import torch as th 
import torch.nn as nn 
import seals , os
from tensorboardX import SummaryWriter
from gymnasium.wrappers.record_video import RecordVideo 
import random 
random.seed(0)
np.random.seed(0)
th.manual_seed(0)
if th.cuda.is_available():
    th.cuda.manual_seed_all(0)
    th.cuda.manual_seed(0)
"""
This is for variable horizon environment(episode)
"""
VARIABLE_HORIZON = False
LOG_DIR = '/home/cai/Desktop/GRU_reward/ensemble_result'
N_ENVS = 8
N_STEPS = int(2048 / N_ENVS)
BATCH_SIZE = 64
TEST_EPI = 10 
RENDER = True
RECODE_TEST_EP = True
RENDER_MODE = 'rgb_array'
CLIP_SIZE = 50 
MAX_EPI_STEP = 500
ENV_NAME = 'FHContinuous_MountainCarEnv-v0' 
HARDCORE = False # it only works BipedalWalker family
INTERACTION_NUM = 60
TOTAL_TIME_STEP = 5_000
TOTAL_COMPARISION = 600
QUEUE_SIZE = 100
OVERSAMPLEING_FACTOR = 5
ENSEMBLE_SIZE=3
"""
ENV_NAME:

    BipedalWalker family: 
        FHBipedalWalker-v1: absorb observation and reward 
        FHBipedalWalker-v2: only has a truncated (i.e. hasn't a termination) and hasn't absorb series
        BipedalWalker-v3: original
        VHBipedalWalker-v1: add step penalty
        
    CartPole
        seals/CartPole-v0: reference https://github.com/HumanCompatibleAI/seals/blob/master/src/seals/classic_control.py
    
    Pendulum
        Pendulum-v1: reference https://imitation.readthedocs.io/en/latest/tutorials/5_train_preference_comparisons.html
    
    Continuous_MountainCarEnv
        FHContinuous_MountainCarEnv-v0: reference https://github.com/HumanCompatibleAI/seals/blob/master/src/seals/classic_control.py
        MountainCarContinuous-v0: original
"""

if RECODE_TEST_EP and RENDER_MODE != 'rgb_array':
    raise ValueError('If you want to record test_epi, then must set render mode to "rgb_array".')

rng = np.random.default_rng(0)

policy_kwargs = dict(
        net_arch = dict(pi = [32,32], vf = [32,32]),
        activation_fn = nn.ReLU,
        features_extractor_class=NormalizeFeaturesExtractor,
        features_extractor_kwargs=dict(normalize_class=RunningNorm),
        )

venv = make_vec_env(ENV_NAME, rng=rng, 
                    n_envs= N_ENVS, 
                    max_episode_steps = MAX_EPI_STEP, 
                    env_make_kwargs = {'hardcore':HARDCORE} if 'BipedalWalker' in ENV_NAME else None
                    )

reward_net = recurrent_net.CustomRewardNet(venv.observation_space, venv.action_space)

reward_net_ensemble =  RecurrentRewardEnsemble(venv.observation_space,
                                            venv.action_space,
                                            members = [reward_net for _ in range(ENSEMBLE_SIZE)]
                                            )
reward_net_ensemble = RecurrentAddSTDRewardWrapper(reward_net_ensemble, 0.0001)
reward_net_ensemble = reward_net_ensemble.to('cuda')
preference_model = recurrent_preference.RecurrentPreferenceModel(reward_net_ensemble, allow_variable_horizon = VARIABLE_HORIZON)

gatherer = recurrent_preference.RecurrentSyntheticGatherer(rng=rng, allow_variable_horizon = VARIABLE_HORIZON)

fragmenter = recurrent_preference.RecurrentRandomFragmenter(
    warning_threshold=0,
    rng=rng,
    allow_variable_horizon = VARIABLE_HORIZON
)
active_fragmenter = recurrent_preference.RecurrentActiveSelectionFragmenter(
    preference_model,
    fragmenter,
    fragment_sample_factor= 3,
)

reward_trainer = recurrent_preference.RecurrentEnsembleTrainer(
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
    learning_rate= 0.0003,
    use_sde = False,
    n_steps=N_STEPS,
    batch_size=BATCH_SIZE,
    ent_coef=0.01,
    clip_range=0.1,
    gae_lambda=0.95,
    gamma=0.97,
    n_epochs=10,
)
trajectory_generator = recurrent_preference.RecurrentAgentTrainer(
    algorithm=agent,
    reward_fn=reward_net_ensemble,
    venv=venv,
    rng=rng,
)
writer_1 = SummaryWriter(os.path.join(LOG_DIR,ENV_NAME,'GRU'))
pref_comparisons = recurrent_preference.RecurrentPreferenceComparisons(
    trajectory_generator,
    reward_net_ensemble,
    num_iterations=INTERACTION_NUM,  
    fragmenter=active_fragmenter,
    preference_gatherer=gatherer,
    reward_trainer=reward_trainer,
    fragment_length=CLIP_SIZE,
    transition_oversampling=OVERSAMPLEING_FACTOR,
    initial_comparison_frac=0.1,
    comparison_queue_size= QUEUE_SIZE,
    allow_variable_horizon=True,
    initial_epoch_multiplier=4,
    query_schedule="hyperbolic",
    tensorboard= writer_1
)
pref_comparisons.train(
    total_timesteps=TOTAL_TIME_STEP,
    total_comparisons=TOTAL_COMPARISION,
)

del pref_comparisons, trajectory_generator, rng, reward_trainer , preference_model, gatherer, fragmenter, venv, reward_net, reward_net_ensemble, active_fragmenter

rng = np.random.default_rng(0)
writer_2 = SummaryWriter(os.path.join(LOG_DIR,ENV_NAME,'Non_GRU'))

venv = make_vec_env(ENV_NAME, rng=rng, 
                    n_envs= N_ENVS, 
                    max_episode_steps = MAX_EPI_STEP, 
                    env_make_kwargs = {'hardcore':HARDCORE} if 'BipedalWalker' in ENV_NAME else None
                    )

reward_net = CustomBasicRewardNet(
    venv.observation_space, venv.action_space
)
reward_net_ensemble = RewardEnsemble(venv.observation_space,
                                            venv.action_space,
                                            [reward_net for _ in range(ENSEMBLE_SIZE)]
                                            )
reward_net_ensemble = AddSTDRewardWrapper(reward_net_ensemble, 0.0001)
reward_net_ensemble = reward_net_ensemble.to('cuda')
gatherer = CustomSyntheticGatherer(rng=rng, allow_variable_horizon = VARIABLE_HORIZON)
preference_model = CustomPreferenceModel(reward_net_ensemble, allow_variable_horizon = VARIABLE_HORIZON)

fragmenter = CustomRandomFragmenter(
    warning_threshold=0,
    rng=rng,
    allow_variable_horizon = VARIABLE_HORIZON
)
active_fragmenter = preference_comparisons.ActiveSelectionFragmenter(
    preference_model,
    fragmenter,
    fragment_sample_factor= 3,
)

reward_trainer = preference_comparisons.EnsembleTrainer(
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
    learning_rate= 0.0003,
    use_sde = False,
    n_steps=N_STEPS,
    batch_size=BATCH_SIZE,
    ent_coef=0.01,
    clip_range=0.1,
    gae_lambda=0.95,
    gamma=0.97,
    n_epochs=10,
)

trajectory_generator = preference_comparisons.AgentTrainer(
    algorithm=agent2,
    reward_fn=reward_net_ensemble,
    venv=venv,
    exploration_frac=0.00,
    rng=rng,
)

pref_comparisons = CustomPreferenceComparisons(
    trajectory_generator,
    reward_net_ensemble,
    num_iterations=INTERACTION_NUM,  # Set to 60 for better performance
    fragmenter=active_fragmenter,
    preference_gatherer=gatherer,
    reward_trainer=reward_trainer,
    fragment_length=CLIP_SIZE,
    transition_oversampling=OVERSAMPLEING_FACTOR,
    initial_comparison_frac=0.1,
    allow_variable_horizon=True,
    comparison_queue_size= QUEUE_SIZE,
    initial_epoch_multiplier=4,
    query_schedule="hyperbolic",
    tensorboard = writer_2
    
)
pref_comparisons.train(
    total_timesteps=TOTAL_TIME_STEP,
    total_comparisons=TOTAL_COMPARISION,
)

del pref_comparisons, trajectory_generator, rng, reward_trainer , preference_model, gatherer, fragmenter, venv, reward_net,  active_fragmenter

import gymnasium as gym 


agents_rewards = []
agents_epi_lenghts = []
for index ,(model, writer) in enumerate(zip([agent, agent2],[writer_1, writer_2])):
    reward_list = []
    epi_lenght_list = []
    if 'BipedalWalker' in ENV_NAME:
        env = gym.make('BipedalWalker-v3', 
                    max_episode_steps = MAX_EPI_STEP, 
                    render_mode = RENDER_MODE,
                    hardcore = HARDCORE
                    )
    else:
        env = gym.make(ENV_NAME, 
            max_episode_steps = MAX_EPI_STEP, 
            render_mode = RENDER_MODE,
            )

    env = RecordVideo(env= env, 
                      video_folder= os.path.join(LOG_DIR, ENV_NAME, f'{VARIABLE_HORIZON}_video'), 
                      name_prefix= 'GRU_reward' if index == 0 else "Non_GRU_reward" 
                        ) if RECODE_TEST_EP else env
    total_reward = 0
    step = 0
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
            if RENDER:
                env.render()
            if terminated or truncated:
                is_done = True 
            writer.add_scalar('eval/cummulated_reward',total_reward, step)
            step += 1 
            total_reward += reward

        reward_list.append(reward_)
        epi_lenght_list.append(epi_lenght)
    del env
    agents_rewards.append([np.mean(reward_list), np.var(reward_list), np.std(reward_list)])
    agents_epi_lenghts.append([np.mean(epi_lenght_list), np.var(epi_lenght_list), np.std(epi_lenght_list)])

print(f'[ENV_NAME]: {ENV_NAME}, [CLIP_SIZE]: {CLIP_SIZE}, [TEST_EPI]: {TEST_EPI}')

print(f'[GRU]\n\
      [reward mean]: {agents_rewards[0][0]}, [reward var]: {agents_rewards[0][1]}, [reward std]: {agents_rewards[0][2]}\n\
        [episode mean]: {agents_epi_lenghts[0][0]}, [episode var]: {agents_epi_lenghts[0][1]}, [episode std]: {agents_epi_lenghts[0][2]}')

print(f'[No GRU]\n\
      [reward mean]: {agents_rewards[1][0]}, [reward var]: {agents_rewards[1][1]}, [reward std]: {agents_rewards[1][2]}\n\
        [episode mean]: {agents_epi_lenghts[1][0]}, [episode var]: {agents_epi_lenghts[1][1]}, [episode std]: {agents_epi_lenghts[1][2]}')