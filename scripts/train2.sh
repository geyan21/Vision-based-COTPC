cd ../src && 

CUDA_VISIBLE_DEVICES=0 python train.py \
   --model_name=LiftCube-V1-test2 \
   --num_traj=100 --n_iters=100_000 \
   --batch_size=12 \
   --context_length=60 --model_type=s+a+cot \
   --task=LiftCube-v1 --key_state_coeff=0.1 \
   --key_state_loss=0 --key_states=ab \
   --init_lr=1e-3 --num_workers=2