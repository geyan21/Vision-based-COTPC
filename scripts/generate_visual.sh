cd /data/geyan21/projects/ManiSkill2-Learn
# LiftCube-v0
# StackCube-v0
# PegInsertionSide-v0

# Replay demonstrations with control_mode=pd_joint_delta_pos
# python -m mani_skill2.trajectory.replay_trajectory --traj-path $path/trajectory.state.pd_joint_delta_pos.h5 \
#   --save-traj --target-control-mode $control_mode --obs-mode pointcloud --num-procs 10

env="PegInsertionSide-v0"
path="/data/geyan21/projects/CoTPC/data/v0/raw/demos/rigid_body/$env"
xvfb-run -a python -m mani_skill2.trajectory.replay_trajectory \
    --traj-path $path/trajectory.h5 \
    --save-traj --target-control-mode pd_joint_delta_pos \
    --obs-mode rgbd --num-procs 20

mkdir -p $path/merged_visual
mv $path/*.rgbd.pd_joint_delta_pos.h5 \
    $path/merged_visual/
mv $path/*.rgbd.pd_joint_delta_pos.json \
    $path/merged_visual/

python -m mani_skill2.trajectory.merge_trajectory \
    -i $path/merged_visual -p *.h5 \
    -o $path/trajectory.rgbd.pd_joint_delta_pos.h5




# env="StackCube-v0"
# path="/data/geyan21/projects/CoTPC/data/v0/raw/demos/rigid_body/$env"
# control_mode="pd_joint_delta_pos"

# # Generate pointcloud demo
# python tools/convert_state.py \
# --env-name=$env \
# --num-procs=12 \
# --traj-name=$path/trajectory.state.pd_joint_delta_pos.h5 \
# --json-name=$path/trajectory.state.pd_joint_delta_pos.json \
# --output-name=$path/trajectory.visual.pd_joint_delta_pos_pointcloud.h5 \
# --control-mode=$control_mode \
# --max-num-traj=-1 \
# --obs-mode=pointcloud \
# --reward-mode=dense \
# --obs-frame=ee \
# --n-points=1200

# # Generate rgbd demo 
# python tools/convert_state.py \
# --env-name=$env \
# --num-procs=12 \
# --traj-name=$path/trajectory.state.pd_joint_delta_pos.h5 \
# --json-name=$path/trajectory.state.pd_joint_delta_pos.json \
# --output-name=$path/trajectory.visual.pd_joint_delta_pos_rgbd.h5 \
# --control-mode=$control_mode \
# --max-num-traj=-1 \
# --obs-mode=rgbd \
# --reward-mode=dense