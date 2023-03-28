#!/bin/bash

# LiftCube-v0
# PushChair-v1
# StackCube-v0
# PegInsertionSide-v0
# TurnFaucet-v0

cd src && 

# task="LiftCube-v0"
# # Example script for PickCube training (with a good set of hyper-parameters). 
# CUDA_VISIBLE_DEVICES=0 xvfb-run -a python train.py \
#    --model_name=$task \
#    --num_traj=500 --n_iters=500_000 \
#    --from_ckpt=200_000 \
#    --context_length=60 --model_type=s+a+cot \
#    --task=$task --key_state_coeff=0.1 \
#    --key_state_loss=0 --key_states=ab \
#    --init_lr=5e-4 --num_workers=20

# task="PegInsertionSide-v0"
# CUDA_VISIBLE_DEVICES=2 xvfb-run -a python train.py \
#    --model_name=$task \
#    --num_traj=500 --n_iters=2000_000 \
#    --from_ckpt=1_000_000 \
#    --context_length=60 --model_type=s+a+cot \
#    --task=$task --key_state_coeff=0.1 \
#    --key_state_loss=0 --key_states=abc \
#    --init_lr=5e-4 --num_workers=10

task="TurnFaucet-v2"
CUDA_VISIBLE_DEVICES=0 xvfb-run -a python train.py \
   --model_name=$task \
   --num_traj=500 --n_iters=2000_000 \
   --context_length=60 --model_type=s+a+cot \
   --task=$task --key_state_coeff=0.1 \
   --key_state_loss=0 --key_states=ab \
   --init_lr=5e-4 --num_workers=10

# task="PushChair-v2"
# # model_name = "PushChair-v2-vision"
# CUDA_VISIBLE_DEVICES=2 xvfb-run -a python train.py \
#    --model_name=$task \
#    --obs_mode='state' \
#    --control_mode='base_pd_joint_vel_arm_pd_joint_vel' \
#    --num_traj=500 --n_iters=2000_000 \
#    --context_length=60 --model_type=s+a+cot \
#    --task=$task --key_state_coeff=0.1 \
#    --key_state_loss=0 --key_states=ab \
#    --init_lr=5e-4 --num_workers=10

# task="StackCube-v0"
# CUDA_VISIBLE_DEVICES=1 xvfb-run -a python train.py \
#    --model_name=$task \
#    --num_traj=500 --n_iters=2000_000 \
#    --from_ckpt=1_000_000 \
#    --context_length=60 --model_type=s+a+cot \
#    --task=$task --key_state_coeff=0.1 \
#    --key_state_loss=0 --key_states=abc \
#    --init_lr=5e-4 --num_workers=10

# CUDA_VISIBLE_DEVICES=0 python train.py \
#     --model_name=some_model_name \
#     --num_traj=500 --n_iters=1_600_000 \
#     --context_length=60 --model_type=s+a+cot \
#     --task=TurnFaucet-v0 --key_state_coeff=0.1 \
#     --key_state_loss=0 --key_states=ab \
#     --init_lr=5e-4 --num_workers=20
