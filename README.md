 Custom imitation library for making a RecurrentReward model. 
 I used GRU. 
 
 

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

## result in MountainCar (w/o variable horizon and w/o ensemble)

[Note]

To make a fixed horizon, i used a ```AbsorbAfterDoneWrapper``` which is from [seals](https://github.com/HumanCompatibleAI/seals/blob/master/src/seals/util.py).

![image](https://github.com/CAI23sbP/RecurrentRLHF/assets/108871750/e64e7635-937c-4f58-bb2d-d78f8d7d54fe)


1. GRU
   


https://github.com/CAI23sbP/RecurrentRLHF/assets/108871750/37374598-a2a4-43c8-9ac3-3cce9b2ad183



2. No GRU 


https://github.com/CAI23sbP/RecurrentRLHF/assets/108871750/ed2f1cf2-7529-4fce-a49b-7a8968ef5587


## result in MountainCar(w/o variable horizon and w/ ensemble)


1. GRU w/ ensemble
   
https://github.com/CAI23sbP/RecurrentRLHF/assets/108871750/6643f5ca-4fa0-4b67-8bff-55dab21ada14



2. No GRU w/ ensemble
   
https://github.com/CAI23sbP/RecurrentRLHF/assets/108871750/4e4e2ea5-5893-4d0a-8c3d-038e11ec250a



## My sister project (related to GRU) ## 

GRU-PPO for stable-baselines3 (or contrib) library

https://github.com/CAI23sbP/GRU_AC/tree/master


