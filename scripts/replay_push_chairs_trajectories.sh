env="PushChair-v1"
path="/data/geyan21/projects/CoTPC/data/v0/raw/demos/rigid_body/$env"
# Assume that the *.h5 and *.json are in `../data/rigid_body_envs/TurnFaucet-v0/raw`,
# replay the trajectories with a subset of a total of 10 faucet models.
for s in 3001 3003 3005 3008 3010 3013 3016 3020 3021 3022 
do
    python -m mani_skill2.trajectory.replay_trajectory \
        --traj-path $path/$s/trajectory.h5 \
	    --save-traj --target-control-mode base_pd_joint_vel_arm_pd_joint_vel \ # different from the original
        --obs-mode state --num-procs 20
done

mkdir -p $path/merged
mv $path/*.state.pd_joint_delta_pos.h5 \
    $path/merged/
mv $path/*.state.pd_joint_delta_pos.json \
    $path/merged/

python -m mani_skill2.trajectory.merge_trajectory \
    -i $path/merged -p *.h5 \
    -o $path/trajectory.state.pd_joint_delta_pos.h5