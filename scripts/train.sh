#!/bin/bash

#cd src && 
cd ../src && 

# # Example script for PickCube training (with a good set of hyper-parameters). 
# CUDA_VISIBLE_DEVICES=1 xvfb-run -a python train.py \
#    --model_name=PickCube_V0 \
#    --num_traj=500 --n_iters=1_600_000 \
#    --context_length=60 --model_type=s+a+cot \
#    --task=PickCube-v0 --key_state_coeff=0.1 \
#    --key_state_loss=0 --key_states=ab \
#    --init_lr=5e-4 --num_workers=20

CUDA_VISIBLE_DEVICES=0 python train.py \
   --model_name=LiftCube-v1-resnet \
   --num_traj=10 --n_iters=100_000 \
   --batch_size=4 \
   --context_length=50 --min_seq_length=50 --model_type=s+a+cot \
   --task=LiftCube-v1 --key_state_coeff=0.1 \
   --key_state_loss=0 --key_states=ab \
   --init_lr=5e-4 --num_workers=0

# CUDA_VISIBLE_DEVICES=0 python train.py \
#     --model_name=some_model_name \
#     --num_traj=500 --n_iters=1_600_000 \
#     --context_length=60 --model_type=s+a+cot \
#     --task=TurnFaucet-v0 --key_state_coeff=0.1 \
#     --key_state_loss=0 --key_states=ab \
#     --init_lr=5e-4 --num_workers=20
