#!/bin/bash
# LiftCube-v0
# PushChair-v1
# StackCube-v0
# PegInsertionSide-v0

# Assume that the *.h5 and *.json are in `../data/rigid_body_envs/TurnFaucet-v0/raw`,
# replay the trajectories with a subset of a total of 10 faucet models.
env="StackCube-v0"
path="/data/geyan21/projects/CoTPC/data/v0/raw/demos/rigid_body/$env"
xvfb-run -a python -m mani_skill2.trajectory.replay_trajectory \
    --traj-path $path/trajectory.h5 \
    --save-traj --target-control-mode pd_joint_delta_pos \
    --obs-mode state --num-procs 20

mkdir -p $path/merged
mv $path/*.state.pd_joint_delta_pos.h5 \
    $path/merged/
mv $path/*.state.pd_joint_delta_pos.json \
    $path/merged/

python -m mani_skill2.trajectory.merge_trajectory \
    -i $path/merged -p *.h5 \
    -o $path/trajectory.state.pd_joint_delta_pos.h5