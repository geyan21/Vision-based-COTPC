#!/bin/bash
# LiftCube-v0 240_000 95.00 480000 96.0 visual 300_000 60.0(100 traj)
# PickCube-v0 680_000 78.60
# StackCube-v0 160_000 14.20 840000 21.8 2000_000 31.2 1000_000 20.2
# PegInsertionSide-v0 240_000 0.00 680_000 3.00 2000_000 6.00 1000_000 5.00
# TurnFaucet-v1 2000_000 13.8 1000_000 12.2
# PushChair-v1 2000_000 8.0 1000_000 8.0
# PushChair-v2 520_000 6.8

# visual 500_000 
# PegInsertionSide-v0 0
# LiftCube-v0 53.0 (500000)
# TurnFaucet-v2 is_contacted 16 success 0 (500000)

cd src && 

# xvfb-run -a python eval_modified.py --eval_max_steps=200 \
#     --from_ckpt=110_000 --task=TurnFaucet-v2 \
#     --model_name=TurnFaucet-v2-visual-doubleCamera


xvfb-run -a python eval_modified.py --eval_max_steps=160 \
    --from_ckpt=110_000 --task=LiftCube-v0 \
    --model_name=LiftCube-v0-visual-hand

xvfb-run -a python eval_modified.py --eval_max_steps=200 \
    --from_ckpt=110_000 --task=PegInsertionSide-v0 \
    --model_name=PegInsertionSide-v0-visual-doubleCamera


# xvfb-run -a python eval.py --num_traj=125 --eval_max_steps=105 \
#     --key_states=ab --key_state_loss=0 \
#     --from_ckpt=1_000_000 --task=PushChair-v1 \
#     --control_mode='base_pd_joint_vel_arm_pd_joint_vel' \
#     --model_name=PushChair-v1

# xvfb-run -a python eval_modified.py --num_traj=500 --eval_max_steps=150 \
#     --key_states=ab --key_state_loss=0 \
#     --from_ckpt=100_000 --task=PushChair-v2 \
#     --control_mode='base_pd_joint_vel_arm_pd_joint_vel' \
#     --model_name=PushChair-v2

# xvfb-run -a python eval.py --num_traj=500 --eval_max_steps=200 \
#     --key_states=abc --key_state_loss=0 \
#     --from_ckpt=1_000_000 --task=StackCube-v0 \
#     --model_name=StackCube-v0

# xvfb-run -a python eval_cpu.py --num_traj=500 --eval_max_steps=200 \
#     --key_states=ab --key_state_loss=0 \
#     --from_ckpt=200_000 --task=LiftCube-v0 \
#     --model_name=LiftCube-v0

# xvfb-run -a python eval.py --num_traj=500 --eval_max_steps=200 \
#     --key_states=ab --key_state_loss=0 \
#     --from_ckpt=680_000 --task=PickCube-v0 \
#     --model_name=PickCube_V0