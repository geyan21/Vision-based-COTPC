import os
import numpy as np
import argparse
from tqdm import tqdm
from collections import defaultdict

from mani_skill2.utils.io_utils import load_json
import torchvision.transforms as transforms
import mani_skill2.envs  # Load ManiSkill2 envs.
import torch  # Load pytorch after maniskill2 to avoid some import error.

from model_modified import GPTConfig, GPTWithCoT

from vec_env import get_mp_envs  # Used for parallel evaluation.
import pdb
from PIL import Image
import random

try:
    # Use might need this for wandb to work due to protobuf issues.
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

    import wandb
    USE_WANDB = True
    PROJECT_NAME = 'CoTPC'  # Please specify the project name.
except ImportError:
    print('Do not use wandb since it is not found.')
    USE_WANDB = False
USE_WANDB = False
# Please specify MODEL_PATH and DATA_PATH (both are base folders) in `path.py`.
MODEL_PATH = '/data/geyan21/projects/CoTPC/models'
DATA_PATH = '/data/geyan21/projects/CoTPC/data/v0/raw/demos/rigid_body'  
device = "cuda:3"


@torch.no_grad()
def predict(model, obs_hist, action_hist, state_hist, t):
    # Please modify this function for model_type other than `s+a+cot`.
    assert model.model_type == 's+a+cot'  

    timesteps = torch.from_numpy(t)[:, None].to(device)
    if not action_hist:  # The first step.
        actions = None
    else:
        actions = torch.stack(action_hist, 1).float().to(device)
    states = torch.stack(state_hist, 1).float().to(device)
    obs = torch.stack(obs_hist, 1).float().to(device)

    # T is the max sequence size; S is the current number of steps.
    B, T = states.shape[0], model.block_size + model.len_key_states
    n_head, S = model.config.n_head, states.shape[1] - 1  # Exclude the init state.

    # Masks for the all-to-all key state query tokens in attention layers.
    # The built-in masks for causal (auto-regressive) tokens are in `model.py`.
    key_state_mask = torch.zeros([B, n_head, T, T], dtype=bool)
    m1 = torch.arange(0, T).repeat(B, 1)
    m2 = torch.ones([B, 1]) * (S * 2 + model.len_key_states)
    m3 = m1 > m2  # Tokens in the future are masked out.
    m3 = m3[:, None, None, :].repeat(1, n_head, model.len_key_states, 1)
    key_state_mask[:, :, :model.len_key_states, :] = m3
    key_state_mask = key_state_mask.to(device)

    preds, _ = model(
        obs, states, timesteps, actions=actions, key_state_mask=key_state_mask)
    return preds[:, -1]  # Only output the last action predictions.


def update(model, obs_hist, action_hist, state_hist, obs, actions, states, t):
    # A function used to update the state and action history.
    # Please change this function for model_type other than `s+a+cot`.
    assert model.model_type == 's+a+cot'  

    actions = torch.from_numpy(actions)
    if len(state_hist) == model.block_size // 2:  # The context buffer is full.
        assert len(action_hist) == model.block_size // 2 - 1
        state_hist = state_hist[1:] + [states]
        action_hist = action_hist[1:] + [actions]
        obs_hist = obs_hist[1:] + [obs]
        t += 1
    else:
        state_hist.append(states)
        action_hist.append(actions)
        obs_hist.append(obs)
    return obs_hist, action_hist, state_hist, t


def convert_observation(task, observation):
        # flattens the original observation by flattening the state dictionaries
        # and combining the rgb and depth images

        # image data is not scaled here and is kept as uint16 to save space
        if task=='PushChair-v1' or task=='PushChair-v2':
            image_obs = observation["image"]
            rgb = image_obs["overhead_camera_1"]["rgb"]
            depth = image_obs["overhead_camera_1"]["depth"]
            # combine the RGB and depth images
            rgbd = np.concatenate([rgb, depth], axis=-1)
            return rgbd
        elif task=='TurnFaucet-v0' or task=='TurnFaucet-v1' or task=='TurnFaucet-v2' or task=='StackCube-v0' or task=='LiftCube-v0':
            # only use the hand camera for now
            image_obs = observation["image"]
            rgb1 = image_obs["base_camera"]["rgb"]
            depth1 = image_obs["base_camera"]["depth"]
            rgb2 = image_obs["hand_camera"]["rgb"]
            depth2 = image_obs["hand_camera"]["depth"]
            # combine the RGB and depth images
            rgbd = np.concatenate([rgb1, depth1, rgb2, depth2], axis=-1)
            return rgbd
        elif task=='PegInsertionSide-v0':
            # only use the base camera for now
            image_obs = observation["image"]
            rgb1 = image_obs["base_camera"]["rgb"]
            depth1 = image_obs["base_camera"]["depth"]
            rgb2 = image_obs["hand_camera"]["rgb"]
            depth2 = image_obs["hand_camera"]["depth"]
            # combine the RGB and depth images
            rgbd = np.concatenate([rgb1, depth1, rgb2, depth2], axis=-1)
            return rgbd
        return rgbd

def convert_state(task, observation):
    # flattens the original observation by flattening the state dictionaries
    # and combining the robot proprioception
    if task=='PushChair-v1' or task=='PushChair-v2':
        base_ang_vel = np.array(observation['agent']['base_ang_vel'])
        base_orientation = np.array(observation['agent']['base_orientation'])
        # check the shape of base_ang_vel and base_orientation, numpy array or number?
        if len(base_ang_vel.shape)!=0:
            base_ang_vel = base_ang_vel[:, None]
            base_orientation = base_orientation[:, None]

        base_pos = np.array(observation['agent']['base_pos'])
        base_vel = np.array(observation['agent']['base_vel'])
        qpos = np.array(observation['agent']['qpos'])
        qvel = np.array(observation['agent']['qvel'])
        state = np.hstack([base_pos, base_orientation, base_vel, base_ang_vel, qpos, qvel])
        return state
    base_pose = np.array(observation['agent']['base_pose'])
    qpos = np.array(observation['agent']['qpos'])
    qvel = np.array(observation['agent']['qvel'])
    tcp_pose = np.array(observation['extra']['tcp_pose'])
    # combine the robot proprioception
    state = np.hstack([base_pose, qpos, qvel, tcp_pose])
    return state

img_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

import matplotlib.pyplot as plt
import torchvision
def save_single_channel_img(img, fname):
    """
    visualize a single channel image and save
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    # threshold
    plt.figure()
    plt.imshow(img, cmap="rainbow")
    plt.tight_layout()
    plt.savefig(fname)

def rgbd_process(rgbds):
    T, H, W, C = rgbds.shape
    if C==4:
        # T H W 4
        rgb = torch.stack([img_transform(Image.fromarray(rgbd[:,:,0:3].astype(np.uint8))) for rgbd in rgbds])
        # depth = torch.from_numpy(rgbds[:,:,:,3] / 2**10).unsqueeze(1)
        depth = torch.from_numpy(rgbds[:,:,:,3]).unsqueeze(1)
        # reshape depth from 160*400 to 128*128
        if H!=128 or W!=128:
            depth = torch.nn.functional.interpolate(depth, size=(128, 128), mode='bilinear', align_corners=True)
        rgbd_transformed = torch.cat([rgb, depth], dim=1).numpy()
    else:
        # only use hand camera and thus only use rgb1 and depth1
        # T H W 8
        rgb1 = torch.stack([img_transform(Image.fromarray(rgbd[:,:,0:3].astype(np.uint8))) for rgbd in rgbds])
        rgb2 = torch.stack([img_transform(Image.fromarray(rgbd[:,:,4:7].astype(np.uint8))) for rgbd in rgbds])
        # depth1 = torch.from_numpy(rgbds[:,:,:,3] / 2**10).unsqueeze(1)
        # depth2 = torch.from_numpy(rgbds[:,:,:,7] / 2**10).unsqueeze(1)
        depth1 = torch.from_numpy(rgbds[:,:,:,3]).unsqueeze(1)
        depth2 = torch.from_numpy(rgbds[:,:,:,7]).unsqueeze(1)
        rgbd_transformed = torch.cat([rgb1, depth1, rgb2, depth2], dim=1).numpy()
    
    
    # save_single_channel_img(rgb[0][0], "rgb.png")
    # save_single_channel_img(depth[0][0], "depth.png")
    # torchvision.utils.save_image(rgb[0], "test_rgb.png")
    # torchvision.utils.save_image(depth[0], "test_depth.png")
    # pdb.set_trace()

    
    return rgbd_transformed


def parse_args():
    parser = argparse.ArgumentParser()

    # Hyper-parameters regarding the demo dataset (used to gather eval_ids)
    parser.add_argument('--task', type=str, default='PickCube-v0', help="Task (env-id) in ManiSkill2.")
    parser.add_argument('--control_mode', type=str, default='pd_joint_delta_pos', 
                        help="Control mode used in envs from ManiSkill2.")
    parser.add_argument('--obs_mode', type=str, default='rgbd', 
                        help="State mode used in envs from ManiSkill2.")
    parser.add_argument("--seed", default=0, type=int,help="Random seed for data spliting.")

    # Hyper-parameters regarding the model.
    parser.add_argument("--model_name", default='', type=str, help="Model name to be loaded.")
    parser.add_argument("--from_ckpt", default=-1, type=int, help="Ckpt of the model to be loaded.")
    
    parser.add_argument("--eval_max_steps", default=200, type=int, help="Max steps allowed in eval.")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    assert args.model_name, 'Should specify --model_name'
    assert args.from_ckpt > 0, 'Should specify --from_ckpt'

    # Load demos to fetch the env. seeds used in training.
    json_path = os.path.join(
        DATA_PATH, f'{args.task}/trajectory.{args.obs_mode}.{args.control_mode}.json')
    json_data = load_json(json_path)
    env_kwargs = json_data["env_info"]["env_kwargs"]
    env_kwargs["obs_mode"] = args.obs_mode
    env_kwargs["control_mode"] = args.control_mode
    np.random.seed(args.seed)

    if USE_WANDB:
        wandb.init(project=PROJECT_NAME, name=f'eval/{args.model_name}', 
                   id=f'wandb_id_{args.model_name}', resume='auto')

    # Number of parallel environments.
    n_env = 1 # 25
    # assert len(eval_ids) % n_env == 0, f'{len(eval_ids)}'
    envs = get_mp_envs(args.task, n_env, **env_kwargs)

    # Load the model.
    path = os.path.join(MODEL_PATH, f'{args.model_name}/{args.from_ckpt}.pth')
    ckpt = torch.load(path)
    print('Loaded ckpt from:', path)

    state_dict_from_ckpt, params = ckpt['model'], ckpt['metadata']
    state_dim = state_dict_from_ckpt['state_encoder.net.0.weight'].shape[1]
    action_dim = state_dict_from_ckpt['action_encoder.net.0.weight'].shape[1]
    max_timestep = state_dict_from_ckpt['global_pos_emb'].shape[1]
    conf = GPTConfig(
        params['context_length'], 
        n_layer=params['n_layer'], 
        n_head=params['n_head'], 
        n_embd=params['n_embd'], 
        model_type=params['model_type'], 
        key_states=params['key_states'],
        key_state_loss=params['key_state_loss'],
        max_timestep=max_timestep,
    )
    model = GPTWithCoT(conf, state_dim=state_dim, action_dim=action_dim).to(device)
    model.load_state_dict(state_dict_from_ckpt, strict=True) 
    model.eval()

    length_all = len(json_data["episodes"])
    if args.task == 'TurnFaucet-v0' or args.task == 'TurnFaucet-v1':
        ids = []
        for i in range(10):  # Hard-code the 10 data splits for permutation.
            t_ids = np.random.permutation(
                length_all // 10)[:params['num_traj'] // 10]
            t_ids += i * length_all // 10
            ids.append(t_ids)
        eval_ids = np.concatenate(ids)
    elif args.task == 'TurnFaucet-v2':
        ids = []
        for i in range(60):  # Hard-code the 60 data splits for permutation.
            t_ids = np.random.permutation(length_all//60)[:params['num_traj']//60]
            t_ids += i*length_all//60
            ids.append(t_ids)
        eval_ids = np.concatenate(ids)
    elif args.task == 'PushChair-v1':
        ids = []
        for i in range(5):  # Hard-code the 10 data splits for permutation.
            t_ids = np.random.permutation(length_all//5)[:params['num_traj']//5]
            t_ids += i*length_all//5
            ids.append(t_ids)
        eval_ids = np.concatenate(ids)
    elif args.task == 'PushChair-v2':
        ids = []
        for i in range(25):  # Hard-code the 10 data splits for permutation.
            t_ids = np.random.permutation(length_all//25)[:params['num_traj']//25]
            t_ids += i*length_all//25
            ids.append(t_ids)
        eval_ids = np.concatenate(ids)
    else:
        eval_ids = np.random.permutation(
            len(json_data["episodes"]))[:params['num_traj']]
        
    assert len(eval_ids) % n_env == 0, f'{len(eval_ids)}'

    # random.shuffle(eval_ids)
    eval_ids = eval_ids[:100]
    #pdb.set_trace()

    metric_dict = defaultdict(lambda: [[] for _ in range(len(eval_ids))])

    for start_idx in tqdm(range(0, len(eval_ids), n_env)):
        reset_args_list = []
        for i in range(start_idx, min(start_idx + n_env, len(eval_ids))):
            reset_kwargs = {'seed': json_data["episodes"][eval_ids[i]]["episode_seed"]}
            reset_args_list.append(reset_kwargs)


        # s = torch.from_numpy(envs.reset(reset_args_list)).float()
        observations = envs.reset(reset_args_list) # B x ...
        rgbd = np.stack([convert_observation(args.task, obs) for obs in observations]) # B H W C
        rgbd = rgbd_process(rgbd) # B C H W
        rgbd = torch.from_numpy(rgbd).float()
        s = np.stack([convert_state(args.task, obs) for obs in observations]) # 
        s = torch.from_numpy(s).float()
        # pdb.set_trace()

        obs_hist, state_hist, action_hist, t = [rgbd], [s], [], np.zeros([n_env])

        for step in range(args.eval_max_steps):
            a = predict(model, obs_hist, action_hist, state_hist, t).cpu().numpy()

            observations, _, _, infos = envs.step(a)
            rgbd = np.stack([convert_observation(args.task, obs) for obs in observations]) # B H W C
            rgbd = rgbd_process(rgbd) # B C H W
            o = torch.from_numpy(rgbd).float()
            s = np.stack([convert_state(args.task, obs) for obs in observations])
            s = torch.from_numpy(s).float()
            
            obs_hist, action_hist, state_hist, t = update(
                model, obs_hist, action_hist, state_hist, o, a, s, t)
            
            # Update metrics.
            for i, info in enumerate(infos):
                j = start_idx + i   
                # You might want to use these additional metrics.         
                if args.task == 'LiftCube-v0':
                    metric_dict['is_grasped'][j].append(info['is_grasped'])
                elif args.task == 'StackCube-v0':
                    metric_dict['is_cubaA_grasped'][j].append(info['is_cubaA_grasped'])
                    metric_dict['is_cubeA_on_cubeB'][j].append(info['is_cubeA_on_cubeB'])
                elif args.task == 'PegInsertionSide-v0':
                    metric_dict['is_grasped'][j].append(info['is_grasped'])
                    metric_dict['pre_inserted'][j].append(info['pre_inserted'])
                elif args.task == 'TurnFaucet-v0' or args.task == 'TurnFaucet-v1' or args.task == 'TurnFaucet-v2':
                    metric_dict['is_contacted'][j].append(info['is_contacted'])
                elif args.task == 'PushChair-v1' or args.task == 'PushChair-v2':
                    metric_dict['chair_close_to_target'][j].append(info['chair_close_to_target'])
                metric_dict['success'][j].append(info['success'])
            
    output_str, output_dict = '', dict()
    for k, v in metric_dict.items():
        v = np.mean([np.any(vv) for vv in v]) * 100
        output_str += f'{k} {v:.2f}, '
        output_dict[k] = v
    output_str = output_str[:-2]
    print(output_str)

    if USE_WANDB: 
        output_dict['n_iter'] = args.from_ckpt
        wandb.log(output_dict)

    # Example eval loop with a single env (not the paralleled VecEnv).
    # import gym
    # env = gym.make('Some inputs here.')
    # s = env.reset()
    # state_hist, action_hist, t = [s], [], np.zeros([1])

    # for step in range(args.eval_max_steps):
    #     a = predict(model, action_hist, state_hist, t).cpu().numpy()[0]
        
    #     s, _, _, info = env.step(a)
    #     s = torch.from_numpy(s).float()[None, :]
    #     a = a[None, :]
        
    #     action_hist, state_hist, t = update(model, action_hist, state_hist, a, s, t)