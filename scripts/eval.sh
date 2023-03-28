#!/bin/bash
# LiftCube-v0 240_000 95.00 480000 96.0
# PickCube-v0 680_000 78.60
# StackCube-v0 160_000 14.20 840000 21.8 2000_000 31.2 1000_000 20.2
# PegInsertionSide-v0 240_000 0.00 680_000 3.00 2000_000 6.00 1000_000 5.00
# TurnFaucet-v1 2000_000 13.8 1000_000 12.2
# PushChair-v1 2000_000 8.0 1000_000 8.0
# PushChair-v2 520_000 6.8

cd src_ori && 

# xvfb-run -a python eval.py --num_traj=500 --eval_max_steps=200 \
#     --key_states=ab --key_state_loss=0 \
#     --from_ckpt=80_000 --task=TurnFaucet-v1 \
#     --model_name=TurnFaucet-v1

# xvfb-run -a python eval.py --num_traj=125 --eval_max_steps=105 \
#     --key_states=ab --key_state_loss=0 \
#     --from_ckpt=80_000 --task=PushChair-v1 \
#     --control_mode='base_pd_joint_vel_arm_pd_joint_vel' \
#     --model_name=PushChair-v1

# xvfb-run -a python eval.py --num_traj=500 --eval_max_steps=150 \
#     --key_states=ab --key_state_loss=0 \
#     --from_ckpt=80_000 --task=PushChair-v2 \
#     --control_mode='base_pd_joint_vel_arm_pd_joint_vel' \
#     --model_name=PushChair-v2

xvfb-run -a python eval.py --num_traj=500 --eval_max_steps=200 \
    --key_states=abc --key_state_loss=0 \
    --from_ckpt=80_000 --task=StackCube-v0 \
    --model_name=StackCube-v0

xvfb-run -a python eval.py --num_traj=500 --eval_max_steps=200 \
    --key_states=abc --key_state_loss=0 \
    --from_ckpt=80_000 --task=PegInsertionSide-v0 \
    --model_name=PegInsertionSide-v0

xvfb-run -a python eval.py --num_traj=500 --eval_max_steps=160 \
    --key_states=ab --key_state_loss=0 \
    --from_ckpt=80_000 --task=LiftCube-v0 \
    --model_name=LiftCube-v0



# xvfb-run -a python eval_cpu.py --num_traj=500 --eval_max_steps=200 \
#     --key_states=ab --key_state_loss=0 \
#     --from_ckpt=200_000 --task=LiftCube-v0 \
#     --model_name=LiftCube-v0

# xvfb-run -a python eval.py --num_traj=500 --eval_max_steps=200 \
#     --key_states=ab --key_state_loss=0 \
#     --from_ckpt=680_000 --task=PickCube-v0 \
#     --model_name=PickCube_V0