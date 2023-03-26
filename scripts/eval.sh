#!/bin/bash

cd ../src && 

python eval.py --num_traj=120 --eval_max_steps=200 \
    --key_states=ab --key_state_loss=0 \
    --from_ckpt=100_000 --task=TurnFaucet-v2 \
    --model_name=TurnFaucet-v2-1