#!/bin/bash

cd ../src && 

# python eval.py --num_traj=200 --eval_max_steps=200 \
#     --key_states=abc --key_state_loss=0 \
#     --from_ckpt=200_000 --task=StackCube-v1 \
#     --model_name=StackCube-v1-1

python eval.py --num_traj=100 --eval_max_steps=200 \
    --key_states=ab --key_state_loss=0 \
    --from_ckpt=150_000 --task=LiftCube-v1 \
    --model_name=LiftCube-v1-2