import os
import gym
import torch
import numpy as np
import argparse
from tqdm import tqdm
from collections import defaultdict

from mani_skill2.utils.io_utils import load_json

import mani_skill2.envs  # To load ManiSkill2 envs.
from model import GPTConfig, GPTWithCoT

from vec_env import get_mp_envs  # Used for parallel evaluation.

# Please specify the model and data path (base folder).
MODEL_PATH = '/home/benny/Desktop/CSE291/Vision-based-COTPC/models'
DATA_PATH = '/home/benny/Desktop/CSE291/Vision-based-COTPC/data/CoTPC-Demos'  


@torch.no_grad()
def predict(model, obs_hist, action_hist, state_hist, t):
    # Please modify this function for model_type other than `s+a+cot`.
    assert model.model_type == 's+a+cot'  

    timesteps = torch.from_numpy(t)[:, None].cuda()
    # timesteps = torch.from_numpy(t)[:, None]
    if not action_hist:  # The first step.
        actions = None
    else:
        actions = torch.stack(action_hist, 1).float().cuda()
        # actions = torch.stack(action_hist, 1).float()
    states = torch.stack(state_hist, 1).float().cuda()
    # states = torch.stack(state_hist, 1).float()
    obs = torch.stack(obs_hist, 1).float().cuda()
    # obs = torch.stack(obs_hist, 1).float()

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
    key_state_mask = key_state_mask.cuda()
    # key_state_mask = key_state_mask
    
    preds, _ = model(obs, 
        states, timesteps, actions=actions, key_state_mask=key_state_mask)
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


def parse_args():
    parser = argparse.ArgumentParser()

    # Hyper-parameters regarding the demo dataset (used to gather eval_ids)
    parser.add_argument('--task', type=str, default='PickCube-v0', help="Task (env-id) in ManiSkill2.")
    parser.add_argument('--control_mode', type=str, default='pd_joint_delta_pos', 
                        help="Control mode used in envs from ManiSkill2.")
    parser.add_argument('--obs_mode', type=str, default='rgbd', 
                        help="State mode used in envs from ManiSkill2.")
    parser.add_argument("--seed", default=0, type=int,help="Random seed for data spliting.")
    parser.add_argument("--num_traj", default=-1, type=int, help="Number of training trajectories.")

    # Hyper-parameters regarding the model.
    parser.add_argument('--context_length', type=int, default=60, 
                        help="Context size of CoTPC.")
    parser.add_argument("--n_layer", default=4, type=int, help="Number of attention layers.")
    parser.add_argument("--n_head", default=8, type=int, help="Number of attention heads.")
    parser.add_argument("--n_embd", default=320, type=int, help="Hidden feature dimension.")
    parser.add_argument("--rgbd_feature_size", default=256, type=int, help="Length of the rgbd feature vector.")
    parser.add_argument("--state_feature_size", default=64, type=int, help="Length of the state feature vector.")
    parser.add_argument('--model_type', type=str, default='s+a+cot', 
                        help="Model type for the CoTPC model (see GPTConfig).")
    parser.add_argument('--key_states', type=str, default='a', 
                        help="Which key states to use (see GPTConfig for the spec. format).")
    parser.add_argument("--key_state_loss", default='', type=str, 
                        help="Features out of what attention layers to use for key state prediction " +
                        "losses (see GPTConfig for the spec. format).")
    parser.add_argument("--model_name", default='', type=str, help="Model name to be loaded.")
    parser.add_argument("--from_ckpt", default=-1, type=int, help="Ckpt of the model to be loaded.")
    
    parser.add_argument("--eval_max_steps", default=200, type=int, help="Max steps allowed in eval.")

    return parser.parse_args()

def process_obs(obs):
    image_obs = obs["image"]
    rgb1 = image_obs["base_camera"]["rgb"]
    depth1 = image_obs["base_camera"]["depth"]
    rgb2 = image_obs["hand_camera"]["rgb"]
    depth2 = image_obs["hand_camera"]["depth"]
    
    rgb1 = rgb1 / 255.0
    rgb2 = rgb2 / 255.0
    depth1 = depth1 / (2**10)
    depth2 = depth2 / (2**10)
    
    rgbd = np.concatenate([rgb1, depth1, rgb2, depth2], axis=-1)
    rgbd = np.transpose(rgbd, (2, 0, 1))
     
    base_pose = np.array(obs['agent']['base_pose'])
    qpos = np.array(obs['agent']['qpos'])
    qvel = np.array(obs['agent']['qvel'])
    tcp_pose = np.array(obs['extra']['tcp_pose'])
    state = np.hstack([base_pose, qpos, qvel, tcp_pose])
    
    return rgbd, state

if __name__ == "__main__":

    args = parse_args()
    assert args.model_name, 'Should specify --model_name'
    assert args.from_ckpt > 0, 'Should specify --from_ckpt'
    assert args.n_embd == args.rgbd_feature_size + args.state_feature_size

    # Load the model.
    path = os.path.join(MODEL_PATH, f'{args.model_name}/{args.from_ckpt}.pth')
    print('Loaded ckpt from:', path)  
    state_dict_from_ckpt = torch.load(path)['model']
    obs_dim = (8, 128, 128) # TODO: hardcode for now
    state_dim = state_dict_from_ckpt['state_encoder.extractors.states.weight'].shape[1]
    action_dim = state_dict_from_ckpt['action_encoder.net.0.weight'].shape[1]
    max_timestep = state_dict_from_ckpt['global_pos_emb'].shape[1]
    print("state_dim: ", state_dim)
    print("action_dim: ", action_dim)
    print("max_timestep: ", max_timestep)
    print("checkpoint: ", args.from_ckpt)
    conf = GPTConfig(
        args.context_length, 
        n_layer=args.n_layer, 
        n_head=args.n_head, 
        n_embd=args.n_embd, 
        model_type=args.model_type, 
        key_states=args.key_states,
        key_state_loss=args.key_state_loss,
        max_timestep=max_timestep,
    )
    model = GPTWithCoT(conf, obs_dim=obs_dim, state_dim=state_dim, action_dim=action_dim,
        rgbd_feature_size=args.rgbd_feature_size, state_feature_size=args.state_feature_size).cuda()
    # model = GPTWithCoT(conf, obs_dim=obs_dim, state_dim=state_dim, action_dim=action_dim,
    #     rgbd_feature_size=args.rgbd_feature_size, state_feature_size=args.state_feature_size)
    model.load_state_dict(state_dict_from_ckpt, strict=False) 
    model.eval()

    # Load demos to fetch the env. seeds used in training.
    json_path = os.path.join(
        DATA_PATH, f'{args.task}/trajectory.{args.obs_mode}.{args.control_mode}.json')
    json_data = load_json(json_path)
    env_kwargs = json_data["env_info"]["env_kwargs"]
    env_kwargs["obs_mode"] = args.obs_mode
    env_kwargs["control_mode"] = args.control_mode
    np.random.seed(args.seed)
    if args.task == 'TurnFaucet-v0':
        length_all = len(json_data["episodes"])
        ids = []
        for i in range(10):  # Hard-code the 10 data splits for permutation.
            t_ids = np.random.permutation(length_all//10)[:args.num_traj//10]
            t_ids += i*length_all//10
            ids.append(t_ids)
        eval_ids = np.concatenate(ids)
    else:
        eval_ids = np.random.permutation(len(json_data["episodes"]))[:args.num_traj]

    # single env
    metric_dict = defaultdict(lambda: [[] for _ in range(len(eval_ids))])
    env = gym.make(args.task, **env_kwargs)
    for start_idx in tqdm(range(0, len(eval_ids))):
        # Example eval loop with a single env (not the paralleled VecEnv).
        reset_kwargs = {'seed': json_data["episodes"][eval_ids[start_idx]]["episode_seed"]}
        env.seed(reset_kwargs['seed'])
        obs = env.reset()
        o, s = process_obs(obs)
        o = torch.from_numpy(o).float()[None, :]
        s = torch.from_numpy(s).float()[None, :]
        obs_hist, state_hist, action_hist, t = [o], [s], [], np.zeros([1])

        # eval_max_steps doesn't work...
        # for step in range(args.eval_max_steps):
        for step in range(max_timestep):
            a = predict(model, obs_hist, action_hist, state_hist, t).cpu().numpy()[0]
            
            obs, _, _, info = env.step(a)
            
            o, s = process_obs(obs)
            o = torch.from_numpy(o).float()[None, :]
            s = torch.from_numpy(s).float()[None, :]
            a = a[None, :]
            
            obs_hist, action_hist, state_hist, t = update(model, obs_hist, action_hist, 
                                                          state_hist, o, a, s, t)

            # Update metrics.
            # You might want to use these additional metrics.         
            if args.task == 'PickCube-v0' or args.task == 'LiftCube-v1':
                metric_dict['is_grasped'][start_idx].append(info['is_grasped'])
            # if args.task == 'StackCube-v0':
            #     metric_dict['is_cubaA_grasped'][j].append(info['is_cubaA_grasped'])
            #     metric_dict['is_cubeA_on_cubeB'][j].append(info['is_cubeA_on_cubeB'])
            # if args.task == 'PegInsertionSide-v0':
            #     metric_dict['is_grasped'][j].append(info['is_grasped'])
            #     metric_dict['pre_inserted'][j].append(info['pre_inserted'])
            # if args.task == 'TurnFaucet-v0':
            #     metric_dict['is_contacted'][j].append(info['is_contacted'])
            metric_dict['success'][start_idx].append(info['success'])

    # # Parallel Environment
    # n_env = 1  # Number of parallel environments.
    # assert len(eval_ids) % n_env == 0, f'{len(eval_ids)}'
    # envs = get_mp_envs(args.task, n_env, **env_kwargs)

    # metric_dict = defaultdict(lambda: [[] for _ in range(len(eval_ids))])

    # for start_idx in tqdm(range(0, len(eval_ids), n_env)):
    #     reset_args_list = []
    #     for i in range(start_idx, min(start_idx + n_env, len(eval_ids))):
    #         reset_kwargs = {'seed': json_data["episodes"][eval_ids[i]]["episode_seed"]}
    #         reset_args_list.append(reset_kwargs)

    #     env_hist = envs.reset(reset_args_list)
    #     # print(env_hist[0].keys())
    #     o = np.zeros([n_env, 8, 128, 128])
    #     s = np.zeros([n_env, state_dim])
    #     for i, env in enumerate(env_hist):
    #         ob, st = process_obs(env)
    #         o[i], s[i] = ob, st
    #     o = torch.from_numpy(o).float()
    #     s = torch.from_numpy(s).float()
    #     obs_hist, state_hist, action_hist, t = [o], [s], [], np.zeros([n_env])

    #     # for step in range(args.eval_max_steps):
    #     for step in range(max_timestep):
    #         print("step: ", step)
    #         a = predict(model, obs_hist, action_hist, state_hist, t).cpu().numpy()

    #         obs, _, _, infos = envs.step(a)
    #         o = np.zeros([n_env, 8, 128, 128])
    #         s = np.zeros([n_env, state_dim])
    #         for i, env in enumerate(obs):
    #             ob, st = process_obs(env)
    #             o[i], s[i] = ob, st
    #         o = torch.from_numpy(o).float()
    #         s = torch.from_numpy(s).float()
            
    #         obs_hist, action_hist, state_hist, t = update(
    #             model, obs_hist, action_hist, state_hist, o, a, s, t)
            
    #         # Update metrics.
    #         for i, info in enumerate(infos):
    #             j = start_idx + i   
    #             # You might want to use these additional metrics.         
    #             if args.task == 'PickCube-v0':
    #                 metric_dict['is_grasped'][j].append(info['is_grasped'])
    #             # if args.task == 'StackCube-v0':
    #             #     metric_dict['is_cubaA_grasped'][j].append(info['is_cubaA_grasped'])
    #             #     metric_dict['is_cubeA_on_cubeB'][j].append(info['is_cubeA_on_cubeB'])
    #             # if args.task == 'PegInsertionSide-v0':
    #             #     metric_dict['is_grasped'][j].append(info['is_grasped'])
    #             #     metric_dict['pre_inserted'][j].append(info['pre_inserted'])
    #             # if args.task == 'TurnFaucet-v0':
    #             #     metric_dict['is_contacted'][j].append(info['is_contacted'])
    #             metric_dict['success'][j].append(info['success'])
    
    output_str = ''
    for k, v in metric_dict.items():
        v = np.mean([np.any(vv) for vv in v]) * 100
        output_str += f'{k} {v:.2f}, '
    output_str = output_str[:-2]
    print(output_str)

    # Example eval loop with a single env (not the paralleled VecEnv).
    # env = gym.make('Some inputs here.')
    # s = env.reset()
    # state_hist, action_hist, t = [s], [], np.zeros([1])

    # for step in range(args.eval_max_steps):
    #     a = predict(model, action_hist, state_hist, t).cpu().numpy()[0]
        
    #     s, _, _, info = env.step(a)
    #     s = torch.from_numpy(s).float()[None, :]
    #     a = a[None, :]
        
    #     action_hist, state_hist, t = update(model, action_hist, state_hist, a, s, t)
