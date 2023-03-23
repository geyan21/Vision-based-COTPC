#!/bin/bash

#cd src && 
cd ../src && 

python train.py \
   --model_name=TurnFaucet-v1-1 \
   --num_traj=200 --n_iters=200_000 \
   --batch_size=12 \
   --context_length=60 --model_type=s+a+cot \
   --task=TurnFaucet-v1 --key_state_coeff=0.1 \
   --key_state_loss=0 --key_states=ab \
   --init_lr=5e-4 --num_workers=4

# CUDA_VISIBLE_DEVICES=0 python train.py \
#     --model_name=some_model_name \
#     --num_traj=500 --n_iters=1_600_000 \
#     --context_length=60 --model_type=s+a+cot \
#     --task=TurnFaucet-v0 --key_state_coeff=0.1 \
#     --key_state_loss=0 --key_states=ab \
#     --init_lr=5e-4 --num_workers=20
