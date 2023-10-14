import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from a2c_ppo_acktr_article import utils
from a2c_ppo_acktr_article.envs import make_vec_envs
from a2c_ppo_acktr_article.action_scaling import scale_action
from a2c_ppo_acktr_article.arguments import get_args


def check_circuit():
    dir_path = '/home/giovani/article/expert/'
    file_name = 'rollout_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt4'

    roll_path = dir_path + file_name
    rollout = torch.load(roll_path)
    obs = rollout['obs'][20]

    fig, ax = plt.subplots(1, 1)
    ax.imshow(obs[0, 0, :, :].cpu().numpy())

    plt.show()


def viz_rollout():
    dir_path = '/home/giovani/article/trained_models/ppo_article/'
    file_name = 'rollout_CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt'

    roll_path = dir_path + file_name

    rollout = torch.load(roll_path)

    acs_th = torch.cat(rollout['acs'])
    acs = acs_th.cpu().numpy()

    fig, ax = plt.subplots(3, 1, figsize=(8, 8))
    fig.suptitle(f'score={rollout["scr"]}')
    t = np.ones_like(acs[:, 0])

    ax[0].plot(acs[:, 0])
    ax[0].plot(t * 1)
    ax[0].plot(t * -1)

    ax[0].set_ylabel('Action[0]')

    ax[1].plot(acs[:, 1])
    ax[1].set_ylabel('Action[1]')
    ax[1].plot(t * 1)
    ax[1].plot(t * -1)
    ax[1].set_xlabel('Time step')

    plt.show()




def save_expert_demo():
    device = 'cuda:0'
    args = get_args()
    path = os.getcwd() + '/article'
    eval_log_dir = path + '/eval'
    expert_dir = path + '/expert/batch_' + args.distribution
    os.makedirs(path, exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)
    os.makedirs(expert_dir, exist_ok=True)
    idx = 2

    dir_path = path
    file_name = 'article/trained_models/agents_beta/CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt'
    seed = 2
    print(f'seed={seed}')

    deterministic = True

    actor_critic, obs_rms, args = torch.load(file_name, map_location=device)
    args.num_processes = 1

    eval_envs = make_vec_envs(args.env_name, seed, args.num_processes,
                              None, eval_log_dir, device, True)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        args.num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(args.num_processes, 1, device=device)

    rollout = {'obs': [],
               'acs': [],
               'rwd': [],
               'scr': []}
    step = 0
    fail_rate = 0
    thresh = 920 if args.env_name == 'CarRacing-v0' else 200

    print(f'... Starting simulation ...')
    n_episodes = 100
    while len(eval_episode_rewards) < n_episodes:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=deterministic)

        rollout['obs'].append(obs.clone().detach())

        rollout['acs'].append(action)


        # action scaling
        # Scale action from 2D to 3D according to distribution
        action_env = scale_action(action, args)

        # Obser reward and next obs
        obs, rwd, done, infos = eval_envs.step(action_env)

        eval_envs.render()

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                print(f'Episode {len(eval_episode_rewards)}')
                eval_episode_rewards.append(info['episode']['r'])
                rollout['scr'].append(info['episode']['r'])
                print(f"score={info['episode']['r']}")
                if rollout['scr'][0] > thresh:
                    rollout_file = 'rollout_j='+str(len(eval_episode_rewards)) + "_"+file_name
                    torch.save(rollout, expert_dir + rollout_file)
                print(f'Episode length = {info["episode"]["l"]}')
                if info['episode']['r'] < thresh:
                    fail_rate += 1
                rollout = {'obs': [],
                           'acs': [],
                           'rwd': [],
                           'scr': []}


    eval_envs.close()

    return file_name, dir_path


def plot_evaluation_CR(file_name: str):
    dir_path = '/home/giovani/article/trained_models/ppo_article/'
    # file_name = 'CarRacing-v0_seed=1_nsteps_=500_d=beta_nup=1249.pt'

    test_file_name = 'test_score_nep=100_' + file_name
    thresh = 900.

    eval_episode_rewards = torch.load(dir_path + test_file_name)
    _, _, args = torch.load(dir_path + file_name)
    fail_rate = 0
    for i in range(100):
        if eval_episode_rewards[i] < thresh:
            fail_rate += 1

    threshold = np.ones_like(eval_episode_rewards) * thresh
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.suptitle(f'Fail rate {fail_rate / len(eval_episode_rewards) * 100}% '
                 f'nsteps={args.num_steps}'
                 f'mean_score={np.mean(eval_episode_rewards[0:100]):.2f}  +/- {np.std(eval_episode_rewards[0:100]):.2f} \n {file_name}')

    ax.plot(eval_episode_rewards, linestyle='', marker='o')
    ax.plot(threshold)
    ax.set_ylim([-100, thresh + 150])
    plt.show()


if __name__ == '__main__':

    file_name, dir_path = save_expert_demo()
    #viz_rollout()
    #check_circuit()