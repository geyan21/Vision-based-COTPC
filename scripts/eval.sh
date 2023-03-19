#!/bin/bash

cd ../src && 

CUDA_LAUNCH_BLOCKING=1 python eval.py --num_traj=100 --eval_max_steps=200 \
    --key_states=ab --key_state_loss=0 \
    --from_ckpt=50_000 --task=LiftCube-v1 \
    --model_name=LiftCube-V1-test
