# env="PushChair-v2"
# path="/data/geyan21/projects/CoTPC/data/v0/raw/demos/rigid_body/$env"
# # Assume that the *.h5 and *.json are in `../data/rigid_body_envs/TurnFaucet-v0/raw`,
# # replay the trajectories with a subset of a total of 10 faucet models.
# mkdir -p $path/merged_visual

# #for v2
# for s in 3001 3003 3005 3008 3010 3013 3016 3020 3021 3022 3024 3025 3027 3030 3031 3032 3038 3045 3047 3050 3051 3063 3070 3071 3076
# do
#     xvfb-run -a python -m mani_skill2.trajectory.replay_trajectory \
#         --traj-path $path/$s/trajectory.h5 \
# 	    --save-traj \
#         --obs-mode rgbd --num-procs 20   
#     mv $path/$s/*.rgbd.base_pd_joint_vel_arm_pd_joint_vel.h5 \
#     $path/merged_visual/$s.rgbd.base_pd_joint_vel_arm_pd_joint_vel.h5
#     mv $path/$s/*.rgbd.base_pd_joint_vel_arm_pd_joint_vel.json \
#     $path/merged_visual/$s.rgbd.base_pd_joint_vel_arm_pd_joint_vel.json
# done

# python -m mani_skill2.trajectory.merge_trajectory \
#     -i $path/merged_visual -p *.h5 \
#     -o $path/trajectory.rgbd.base_pd_joint_vel_arm_pd_joint_vel.h5

# # for debug
# xvfb-run -a python -m mani_skill2.trajectory.replay_trajectory \
#         --traj-path $path/3001/trajectory.h5 \
# 	    --save-traj \
#         --max-retry 0 \
#         --obs-mode state --num-procs 10

# for v1:
env="PushChair-v1"
path="/data/geyan21/projects/CoTPC/data/v0/raw/demos/rigid_body/$env"
# Assume that the *.h5 and *.json are in `../data/rigid_body_envs/TurnFaucet-v0/raw`,
# replay the trajectories with a subset of a total of 10 faucet models.
mkdir -p $path/merged_visual

#for v2
for s in 3003 3013 3020 3030 3063
do
    xvfb-run -a python -m mani_skill2.trajectory.replay_trajectory \
        --traj-path $path/$s/trajectory.h5 \
	--save-traj \
        --obs-mode rgbd --num-procs 20   
    mv $path/$s/*.rgbd.base_pd_joint_vel_arm_pd_joint_vel.h5 \
    $path/merged_visual/$s.rgbd.base_pd_joint_vel_arm_pd_joint_vel.h5
    mv $path/$s/*.rgbd.base_pd_joint_vel_arm_pd_joint_vel.json \
    $path/merged_visual/$s.rgbd.base_pd_joint_vel_arm_pd_joint_vel.json
done

python -m mani_skill2.trajectory.merge_trajectory \
    -i $path/merged_visual -p *.h5 \
    -o $path/trajectory.rgbd.base_pd_joint_vel_arm_pd_joint_vel.h5