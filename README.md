 Custom imitation library for making a RecurrentReward model. 
 I used GRU. 
 
 [Note] We apply a variable horizon setting for a robotic task 
 

## How to Train ## 

No Ensembling ```python3 train.py``` 
Ensemblign ```python3 train_ensemble.py```

## setting your reward network ##

go to ```test_net``` and modify your code

## Test env list ##

CartPole

BipedalWalker

Pendulum-v1

MountainCar

## library compatibility  ## 

torch: 1.13.1+cu116

imitation: 1.0.0  

stable_baselines3: 2.3.0

## result in MountainCar(w/ variable horizon)

1. GRU
   


https://github.com/CAI23sbP/RecurrentRLHF/assets/108871750/37374598-a2a4-43c8-9ac3-3cce9b2ad183



2. No GRU 


https://github.com/CAI23sbP/RecurrentRLHF/assets/108871750/ed2f1cf2-7529-4fce-a49b-7a8968ef5587

## My sister project (related to GRU) ## 

GRU-PPO for stable-baselines3 (or contrib) library

https://github.com/CAI23sbP/GRU_AC/tree/master


