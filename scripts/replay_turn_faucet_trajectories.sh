#!/bin/bash
# for v2 visual
# env="TurnFaucet-v2"
# path="/data/geyan21/projects/CoTPC/data/v0/raw/demos/rigid_body/$env"
# # Assume that the *.h5 and *.json are in `../data/rigid_body_envs/TurnFaucet-v0/raw`,
# # replay the trajectories with a subset of a total of 10 faucet models.
# for s in 5005 5006 5007 5010 5011 5012 5014 5015 5016 5018 5020 5021 5023 5024 5025 5027 5028 5029 5030 5033 5034 5035 5037 5038 5039 5040 5041 5043 5045 5046 5047 5048 5049 5050 5051 5052 5053 5055 5056 5057 5058 5060 5061 5062 5063 5064 5065 5067 5068 5069 5070 5072 5073 5075 5076
# do
#     python -m mani_skill2.trajectory.replay_trajectory \
#         --traj-path $path/$s.h5 \
# 	    --save-traj --target-control-mode pd_joint_delta_pos \
#         --obs-mode rgbd --num-procs 20
# done

# mkdir -p $path/merged_visual
# mv $path/*.rgbd.pd_joint_delta_pos.h5 \
#     $path/merged_visual/
# mv $path/*.rgbd.pd_joint_delta_pos.json \
#     $path/merged_visual/

# python -m mani_skill2.trajectory.merge_trajectory \
#     -i $path/merged_visual -p *.h5 \
#     -o $path/trajectory.rgbd.pd_joint_delta_pos.h5

# # for v2 state
# env="TurnFaucet-v2"
# path="/data/geyan21/projects/CoTPC/data/v0/raw/demos/rigid_body/$env"
# # Assume that the *.h5 and *.json are in `../data/rigid_body_envs/TurnFaucet-v0/raw`,
# # replay the trajectories with a subset of a total of 10 faucet models.
# for s in 5004 5005 5006 5007 5010 5011 5012 5014 5015 5016 5018 5020 5021 5023 5024 5025 5027 5028 5029 5030 5033 5034 5035 5037 5038 5039 5040 5041 5043 5045 5046 5047 5048 5049 5050 5051 5052 5053 5055 5056 5057 5058 5060 5061 5062 5063 5064 5065 5067 5068 5069 5070 5072 5073 5075 5076
# do
#     python -m mani_skill2.trajectory.replay_trajectory \
#         --traj-path $path/$s.h5 \
# 	    --save-traj --target-control-mode pd_joint_delta_pos \
#         --obs-mode state --num-procs 10
# done

# mkdir -p $path/merged
# mv $path/*.state.pd_joint_delta_pos.h5 \
#     $path/merged/
# mv $path/*.state.pd_joint_delta_pos.json \
#     $path/merged/

# python -m mani_skill2.trajectory.merge_trajectory \
#     -i $path/merged -p *.h5 \
#     -o $path/trajectory.state.pd_joint_delta_pos.h5

# for v1
# env="TurnFaucet-v1"
# path="/data/geyan21/projects/CoTPC/data/v0/raw/demos/rigid_body/$env"
# # Assume that the *.h5 and *.json are in `../data/rigid_body_envs/TurnFaucet-v0/raw`,
# # replay the trajectories with a subset of a total of 10 faucet models.
# for s in 5002 5021 5023 5028 5029 5045 5047 5051 5056 5063
# do
#     python -m mani_skill2.trajectory.replay_trajectory \
#         --traj-path $path/$s.h5 \
# 	    --save-traj --target-control-mode pd_joint_delta_pos \
#         --obs-mode rgbd --num-procs 5
# done

# mkdir -p $path/merged_visual
# mv $path/*.rgbd.pd_joint_delta_pos.h5 \
#     $path/merged_visual/
# mv $path/*.rgbd.pd_joint_delta_pos.json \
#     $path/merged_visual/

# python -m mani_skill2.trajectory.merge_trajectory \
#     -i $path/merged_visual -p *.h5 \
#     -o $path/trajectory.rgbd.pd_joint_delta_pos.h5


# env="TurnFaucet-v1"
# env1="TurnFaucet-v2"
path="/data/geyan21/projects/CoTPC/data/v0/raw/demos/rigid_body/"
path1="/data/geyan21/projects/CoTPC/backdata"
# mkdir -p $path/$env/merged_visual
# for s in 5002 5021 5023 5028 5029 5045 5047 5051 5056 5063
# do
#     cp $path/$env1/merged_visual/$s.rgbd.pd_joint_delta_pos.h5 \
#         $path/$env/merged_visual/
#     cp $path/$env1/merged_visual/$s.rgbd.pd_joint_delta_pos.json \
#         $path/$env/merged_visual/
# done

for env in "PushChair-v1" "PushChair-v2"
do
    mkdir -p $path1/$env/
    cp $path/$env/trajectory.rgbd.base_pd_joint_vel_arm_pd_joint_vel.h5 \
        $path1/$env/
    cp $path/$env/trajectory.rgbd.base_pd_joint_vel_arm_pd_joint_vel.json \
        $path1/$env/
done