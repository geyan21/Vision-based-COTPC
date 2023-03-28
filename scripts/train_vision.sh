cd src && 

# task="StackCube-v0"
# model_name="StackCube-v0-visual-hand"
# CUDA_VISIBLE_DEVICES=1 python train_modified.py \
#    --model_name=$model_name \
#    --obs_mode='rgbd' \
#    --num_traj=400 --n_iters=501_000 \
#    --context_length=60 --model_type=s+a+cot \
#    --task=$task --key_state_coeff=0.1 \
#    --key_state_loss=0 --key_states=abc \
#    --init_lr=5e-4 --num_workers=10 \
#    --n_embd=832 \
#    --batch_size=1 \
#    --multiplier=10

task="PegInsertionSide-v0"
model_name="PegInsertionSide-v0-visual-hand"
CUDA_VISIBLE_DEVICES=2 python train_modified.py \
   --model_name=$model_name \
   --obs_mode='rgbd' \
   --num_traj=500 --n_iters=1001_000 \
   --from_ckpt=500_000 \
   --context_length=60 --model_type=s+a+cot \
   --task=$task --key_state_coeff=0.1 \
   --key_state_loss=0 --key_states=abc \
   --init_lr=5e-4 --num_workers=10 \
   --n_embd=832 \
   --batch_size=1 \
   --multiplier=10

# task="LiftCube-v0"
# model_name="LiftCube-v0-visual-hand"
# CUDA_VISIBLE_DEVICES=3 python train_modified.py \
#    --model_name=$model_name \
#    --obs_mode='rgbd' \
#    --num_traj=500 --n_iters=1001_000 \
#    --from_ckpt=500_000 \
#    --context_length=60 --model_type=s+a+cot \
#    --task=$task --key_state_coeff=0.1 \
#    --key_state_loss=0 --key_states=ab \
#    --init_lr=5e-4 --num_workers=10 \
#    --n_embd=832 \
#    --batch_size=1 \
#    --log_every=1000 \
#    --multiplier=20

# task="TurnFaucet-v2"
# model_name="TurnFaucet-v2-visual"
# CUDA_VISIBLE_DEVICES=1 python train_modified.py \
#    --model_name=$model_name \
#    --obs_mode='rgbd' \
#    --num_traj=400 --n_iters=1001_000 \
#    --from_ckpt=500_000 \
#    --context_length=60 --model_type=s+a+cot \
#    --task=$task --key_state_coeff=0.1 \
#    --key_state_loss=0 --key_states=ab \
#    --init_lr=5e-4 --num_workers=10 \
#    --n_embd=832 \
#    --batch_size=1 \
#    --log_every=1000 \
#    --multiplier=10

# task="PushChair-v1"
# model_name="PushChair-v1-visual"
# CUDA_VISIBLE_DEVICES=2 python train_modified.py \
#    --model_name=$model_name \
#    --obs_mode='rgbd' \
#    --control_mode='base_pd_joint_vel_arm_pd_joint_vel' \
#    --num_traj=500 --n_iters=501_000 \
#    --context_length=60 --model_type=s+a+cot \
#    --task=$task --key_state_coeff=0.1 \
#    --key_state_loss=0 --key_states=ab \
#    --init_lr=5e-4 --num_workers=10 \
#    --n_embd=832 \
#    --batch_size=1 \
#    --multiplier=10

# task="PushChair-v2"
# model_name="PushChair-v2-visual"
# CUDA_VISIBLE_DEVICES=2 python train_modified.py \
#    --model_name=$model_name \
#    --obs_mode='rgbd' \
#    --control_mode='base_pd_joint_vel_arm_pd_joint_vel' \
#    --num_traj=500 --n_iters=501_000 \
#    --context_length=60 --model_type=s+a+cot \
#    --task=$task --key_state_coeff=0.1 \
#    --key_state_loss=0 --key_states=ab \
#    --init_lr=5e-4 --num_workers=10 \
#    --n_embd=832 \
#    --batch_size=1 \
#    --multiplier=10