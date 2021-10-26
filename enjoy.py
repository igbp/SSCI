import argparse
import os
# workaround to unpickle olf model files
import sys

import numpy as np
import torch

from a2c_ppo_acktr_article.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr_article.utils import get_render_func, get_vec_normalize
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append('a2c_ppo_acktr')

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--env-name',
        default='LunarLanderContinuous-v2',
        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument(
        '--load-dir',
        default='./trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--non-det',
        action='store_true',
        default=False,
        help='whether to use a non-deterministic policy')
    args = parser.parse_args()

    args.det = not args.non_det

    return args


def enjoy():

    args = get_args()

    env = make_vec_envs(
        args.env_name,
        args.seed + 1000,
        1,
        None,
        None,
        device='cpu',
        allow_early_resets=False)

    # Get a render function
    render_func = get_render_func(env)

    # We need to use the same statistics for normalization as used in training
    file_name = 'ppo/'+args.env_name +'nupdates=1249'+ ".pt"

    actor_critic, obs_rms = \
                torch.load(os.path.join(args.load_dir, file_name),
                            map_location='cpu')

    vec_norm = get_vec_normalize(env)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms

    recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
    masks = torch.zeros(1, 1)

    obs = env.reset()

    if render_func is not None:
        render_func('human')

    if args.env_name.find('Bullet') > -1:
        import pybullet as p

        torsoId = -1
        for i in range(p.getNumBodies()):
            if (p.getBodyInfo(i)[0].decode() == "torso"):
                torsoId = i

    while True:
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(
                obs, recurrent_hidden_states, masks, deterministic=args.det)

        # Obser reward and next obs
        obs, reward, done, _ = env.step(action)

        masks.fill_(0.0 if done else 1.0)

        if render_func is not None:
            render_func('human')




def check_std():
    args = get_args()

    env = make_vec_envs(
        args.env_name,
        args.seed + 1000,
        1,
        None,
        None,
        device='cpu',
        allow_early_resets=False)

    # Get a render function
    render_func = get_render_func(env)

    # We need to use the same statistics for normalization as used in training
    file_name = 'ppo/' + 'LunarLanderContinuous-v2nupdates=975.pt'

    actor_critic, obs_rms = \
                torch.load(os.path.join(args.load_dir, file_name),
                            map_location='cpu')

    vec_norm = get_vec_normalize(env)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms

    recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
    masks = torch.zeros(1, 1)

    obs = env.reset()

    if render_func is not None:
        render_func('human')

    if args.env_name.find('Bullet') > -1:
        import pybullet as p

        torsoId = -1
        for i in range(p.getNumBodies()):
            if (p.getBodyInfo(i)[0].decode() == "torso"):
                torsoId = i

    n_samples = 1000
    acs_hist = torch.zeros((n_samples, 2))

    args.det = True
    for i in range(n_samples):
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(
                obs, recurrent_hidden_states, masks, deterministic=args.det)

            acs_hist[i, :] = action[0, :]

    # Obser reward and next obs
        obs, reward, done, _ = env.step(action)

        masks.fill_(0.0 if done else 1.0)

        if render_func is not None:
            render_func('human')




if __name__ =='__main__':
    #enjoy()

    check_std()



