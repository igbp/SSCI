import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams
from stable_baselines3.common.atari_wrappers import (ClipRewardEnv,
                                                     EpisodicLifeEnv,
                                                     FireResetEnv,
                                                     MaxAndSkipEnv,
                                                     NoopResetEnv, WarpFrame)


def evaluate_CR(distr):

    # Env setup
    eval_env = gym.make('CarRacing-v0')
    eval_env = WarpFrame(eval_env)

    eval_env.seed(3)
    device = 'cuda:0'
    obs = eval_env.reset()
    obs_pth = torch.Tensor(obs).to(device)
    n_stack = 4

    obs4frame = torch.cat([obs_pth for _ in range(n_stack)], dim=2)
    obs4frame = obs4frame.to(device)
    obs4frame = obs4frame.permute(2, 0, 1)
    obs4frame = obs4frame.unsqueeze(0)
    print(obs4frame.size())

    # model setup
    file_name = 'CarRacing-v0_seed=1_nsteps_=500_d='+distr+'_nup=1249.pt'
    rollout_file = 'rollout_' + file_name
    dir_path = '/home/giovani/article/trained_models/ppo/'

    actor_critic, _, _ = torch.load(dir_path + file_name)


    eval_recurrent_hidden_states = torch.zeros(
        1, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(1, 1, device=device)
    deterministic = True
    score = 0
    score_hist = [score]
    done = False

    rollout_single = {'obs': [],
                      'obs_pth':[],
                      'rwd': [],
                      'scr': [],
                      'action': []}

    while not done:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs4frame,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=deterministic)

        # action interface
        if 'beta' in file_name:
            action = action * torch.Tensor([2]).to(device) + torch.Tensor([-1., -1.]).to(device)

        acs_pth = torch.zeros([3])
        acs_pth[0] = action[0][0]
        acs_pth[1] = torch.relu(action[0][1])
        acs_pth[2] = torch.relu(-action[0][1])
        acs = acs_pth.cpu().numpy()

        # env step
        obs, rwd, done, info = eval_env.step(acs)

        # rollout log
        score += rwd
        rollout_single['rwd'].append(rwd)
        rollout_single['scr'].append(score)
        rollout_single['obs'].append(obs)
        rollout_single['obs_pth'].append(obs4frame.clone())
        rollout_single['action'].append(action)


        # renders environment
        eval_env.render()

        # prepare next obs
        obs4frame[:, :-1, :, :] = obs4frame[:, 1:, :, :].clone()
        obs_pth = torch.Tensor(obs).to(device)
        obs4frame[:, -1, :, :] = obs_pth[:, :, 0].clone()

    rollout_file = 'rollout_'+file_name
    torch.save(rollout_single, dir_path + rollout_file)
    eval_env.close()



def evaluate_CR_nomerge(distr):

    # Env setup
    eval_env = gym.make('CarRacing-v0')
    eval_env = WarpFrame(eval_env)

    eval_env.seed(3)
    device = 'cuda:0'
    obs = eval_env.reset()
    obs_pth = torch.Tensor(obs).to(device)
    n_stack = 4

    obs4frame = torch.cat([obs_pth for _ in range(n_stack)], dim=2)
    obs4frame = obs4frame.to(device)
    obs4frame = obs4frame.permute(2, 0, 1)
    obs4frame = obs4frame.unsqueeze(0)
    print(obs4frame.size())

    # model setup
    file_name = 'CarRacing-v0_seed=3_nsteps_=500_d=normal_nup=1000.pt'
    rollout_file = 'rollout_' + file_name
    dir_path = '/home/giovani/article/trained_models/ppo/'

    actor_critic, _, _ = torch.load(dir_path + file_name)


    eval_recurrent_hidden_states = torch.zeros(
        1, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(1, 1, device=device)
    deterministic = True
    score = 0
    score_hist = [score]
    done = False

    rollout_single = {'obs': [],
                      'obs_pth': [],
                      'rwd': [],
                      'scr': [],
                      'action': []}

    while not done:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs4frame,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=deterministic)

        # action interface

        if 'beta' in file_name:
            action = action * torch.Tensor([2]).to(device) + torch.Tensor([-1., -1.]).to(device)

        acs_pth = torch.zeros([3])
        acs_pth[0] = action[0][0]
        acs_pth[1] = torch.relu(action[0][1])
        acs_pth[2] = torch.relu(-action[0][1])
        acs = acs_pth.cpu().numpy()


        # env step
        #obs, rwd, done, info = eval_env.step(acs)
        obs, rwd, done, info = eval_env.step(action[0].cpu().numpy())
        print(action[0].cpu().numpy())

        # rollout log
        score += rwd
        rollout_single['rwd'].append(rwd)
        rollout_single['scr'].append(score)
        rollout_single['obs'].append(obs)
        rollout_single['obs_pth'].append(obs4frame.clone())
        rollout_single['action'].append(action)


        # renders environment
        eval_env.render()

        # prepare next obs
        obs4frame[:, :-1, :, :] = obs4frame[:, 1:, :, :].clone()
        obs_pth = torch.Tensor(obs).to(device)
        obs4frame[:, -1, :, :] = obs_pth[:, :, 0].clone()

    rollout_file = 'rollout_'+file_name
    torch.save(rollout_single, dir_path + rollout_file)
    eval_env.close()


def load_conf():
    fontsize = 20
    rcParams['font.family'] = 'serif'
    rcParams['font.sans-serif'] = ['Times New Roman']
    rcParams['font.size'] = fontsize
    rcParams['mathtext.fontset'] = 'stix'
    rcParams['axes.titlesize'] = fontsize
    rcParams['axes.labelsize'] = fontsize
    rcParams['text.usetex'] = True


def compare_CR():
    load_conf()
    distributions = ['normal', 'beta']
    rollouts = []
    dir_path = '/home/giovani/article/trained_models/ppo_article/case_study/'

    fig, ax = plt.subplots(1, 3, figsize=(12,6))
    base = []

    # chart [0, 0]
    for distr in distributions:
        file_name = 'CarRacing-v0_seed=2_nsteps_=500_d=' + distr + '_nup=1249.pt'
        rollout_file = 'rollout_' + file_name

        rollout = torch.load(dir_path + rollout_file)
        rollouts.append(rollout)
        ax[0].plot(rollout['scr'], label=distr)
        if len(rollout['scr']) > len(base):
            base = rollout['scr']

    threshold = np.ones_like(base)*900.0
    ax[0].plot(threshold, linestyle='--', color='red')
    ax[0].legend()
    ax[0].set_xlabel('Environment time steps')
    ax[0].set_ylabel('Reward')

    img_step = 100
    # chart [0, 1]
    ax[1].imshow(rollouts[0]['obs'][img_step], cmap='gray')
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    ax[1].set_title(distributions[0])

    # chart [0, 2]
    ax[2].imshow(rollouts[1]['obs'][img_step], cmap='gray')
    ax[2].get_xaxis().set_visible(False)
    ax[2].get_yaxis().set_visible(False)
    ax[2].set_title(distributions[1])

    plt.show()


def compare_CR_acs():
    load_conf()
    distributions = ['normal', 'beta']
    rollouts = []
    dir_path = '/home/giovani/article/trained_models/ppo_article/case_study/'

    fig, ax = plt.subplots(2, 1, figsize=(12, 6))
    base = []

    # chart [0, 0]
    for distr in distributions:
        file_name = 'CarRacing-v0_seed=2_nsteps_=500_d=' + distr + '_nup=1249.pt'
        rollout_file = 'rollout_' + file_name

        rollout = torch.load(dir_path + rollout_file)
        rollouts.append(rollout)
        #ax[0].plot(rollout['scr'], label=distr)
        if len(rollout['scr']) > len(base):
            base = rollout['scr']

    normal_acs = torch.cat(rollouts[0]['action'])[: ,0].clone()


    bound = torch.ones_like(normal_acs)
    normal_acs = torch.min(normal_acs, bound)
    normal_acs = torch.max(normal_acs, -bound)

    ax[0].plot(normal_acs.cpu().numpy(), linestyle='-', color='blue')
    #ax[0].plot(bound.cpu().numpy(), linestyle='-', color='red')
    #ax[0].plot(-bound.cpu().numpy(), linestyle='-', color='red')

    ax[0].set_title(distributions[0])

    ax[1].plot(torch.cat(rollouts[1]['action'])[:, 0].cpu().numpy(), linestyle='-', color='blue')
    ax[1].set_title(distributions[1])

    plt.show()


def find_observation():
    load_conf()
    distributions = ['normal', 'beta']
    rollouts = []
    models = []
    dir_path = '/home/giovani/article/trained_models/ppo_article/'


    # chart [0, 0]
    for distr in distributions:
        file_name = 'CarRacing-v0_seed=2_nsteps_=500_d=' + distr + '_nup=1249.pt'
        rollout_file = 'rollout_' + file_name

        rollout = torch.load(dir_path + 'case_study/'+rollout_file)
        rollouts.append(rollout)

        model = torch.load(dir_path + file_name)
        models.append(model)


    fig, ax = plt.subplots(5, 5, figsize=(12, 6))
    base = []
    starting_frame = 65
    for i in range(25):
        ax[i//5, i % 5].imshow(rollouts[0]['obs'][starting_frame+i])

    fig.suptitle(f'Chart starting on {starting_frame}')
    plt.show()


def sample_distribution():
    load_conf()
    distributions = ['normal', 'beta']
    rollouts = []
    models = []
    dir_path = '/home/giovani/article/trained_models/ppo_article/'

    # chart [0, 0]
    for distr in distributions:
        file_name = 'CarRacing-v0_seed=2_nsteps_=500_d=' + distr + '_nup=1249.pt'
        rollout_file = 'rollout_' + file_name

        rollout = torch.load(dir_path+'case_study/'+rollout_file)
        rollouts.append(rollout)

        model = torch.load(dir_path + file_name)
        models.append(model)

    starting_frame = 71
    obs = rollouts[0]['obs_pth'][starting_frame].clone()

    actor_critic,_ , _ = models[0]
    # Model params
    device= 'cuda:0'
    eval_recurrent_hidden_states = torch.zeros(
        1, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(1, 1, device=device)
    deterministic = False

    n_samples = 5000
    binwidth = 0.01
    fig, ax = plt.subplots(2, 3, figsize=(20, 6))

    gs = ax[0, 0].get_gridspec()
    # remove the underlying axes
    for a in ax[0:, 0]:
        a.remove()
    axbig = fig.add_subplot(gs[0:, 0])
    axbig.imshow(rollouts[0]['obs'][starting_frame], cmap='gray')
    axbig.get_xaxis().set_visible(False)
    axbig.get_yaxis().set_visible(False)

    for m in range(len(distributions)):
        actor_critic, _, _, = models[m]
        acs = []
        for i in range(n_samples):
            with torch.no_grad():
                _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                    obs,
                    eval_recurrent_hidden_states,
                    eval_masks,
                    deterministic=False)
            if m == 1:
                action = action.clone()*torch.Tensor([2.]).to(device) + torch.Tensor([-1, -1]).to(device)
            acs.append(action.clone())

        acs = torch.cat(acs).cpu().numpy()
        h, _, _ = ax[m, 1].hist(acs[:, 0], bins=np.arange(-3., 3. + binwidth, binwidth), color='blue')
        ax[m, 1].fill_betweenx([0, h.max()], -1., 0., color='yellow', alpha=0.1, label ='Turn left')
        ax[m, 1].fill_betweenx([0, h.max()],  0., 1., color='orange', alpha=0.1, label='Turn right')
        h, _, _ = ax[m, 1].hist(acs[:, 0], bins=np.arange(-3., 3. + binwidth, binwidth), color='blue')
        ax[m, 1].set_xticks([-2, -1, 0, 1, 2])

        h, _,_ = ax[m, 2].hist(acs[:, 1], bins=np.arange(-3., 3. + binwidth, binwidth))
        ax[m, 2].fill_betweenx([0, h.max()], -1., 0., color='lightsalmon', alpha=0.1, label= 'Brake')
        ax[m, 2].fill_betweenx([0, h.max()],  0., 1., color='lightgreen', alpha=0.1, label='Throttle')
        h, _,_ = ax[m, 2].hist(acs[:, 1], bins=np.arange(-3., 3. + binwidth, binwidth), color='blue')
        ax[m, 2].set_xticks([-2, -1, 0 ,1, 2])

    ax[0, 1].legend()
    ax[0, 2].legend()

    ax[0, 1].set_ylabel('Gaussian')
    ax[1, 1].set_ylabel('Beta')

    plt.show()



def sample_distribution2():
    load_conf()
    distributions = ['normal', 'beta']
    rollouts = []
    models = []
    dir_path = '/home/giovani/article/trained_models/ppo_article/'

    # chart [0, 0]
    for distr in distributions:
        file_name = 'CarRacing-v0_seed=2_nsteps_=500_d=' + distr + '_nup=1249.pt'
        rollout_file = 'rollout_' + file_name

        rollout = torch.load(dir_path+'case_study/'+rollout_file)
        rollouts.append(rollout)

        model = torch.load(dir_path + file_name)
        models.append(model)

    starting_frame = 71
    obs = rollouts[0]['obs_pth'][starting_frame].clone()

    actor_critic,_ , _ = models[0]
    # Model params
    device= 'cuda:0'
    eval_recurrent_hidden_states = torch.zeros(
        1, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(1, 1, device=device)
    deterministic = False

    n_samples = 5000
    binwidth = 0.01
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))

    gs = ax[0, 0].get_gridspec()
    # remove the underlying axes

    for m in range(len(distributions)):
        actor_critic, _, _, = models[m]
        acs = []
        for i in range(n_samples):
            with torch.no_grad():
                _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                    obs,
                    eval_recurrent_hidden_states,
                    eval_masks,
                    deterministic=False)
            if m == 1:
                action = action.clone()*torch.Tensor([2.]).to(device) + torch.Tensor([-1, -1]).to(device)
            acs.append(action.clone())

        acs = torch.cat(acs).cpu().numpy()
        h, _, _ = ax[m, 0].hist(acs[:, 0], bins=np.arange(-3., 3. + binwidth, binwidth), color='blue')
        ax[m, 0].fill_betweenx([0, h.max()], -1., 0., color='yellow', alpha=0.1, label ='Turn left')
        ax[m, 0].fill_betweenx([0, h.max()],  0., 1., color='orange', alpha=0.1, label='Turn right')
        h, _, _ = ax[m, 0].hist(acs[:, 0], bins=np.arange(-3., 3. + binwidth, binwidth), color='blue')
        ax[m, 0].set_xticks([-2, -1, 0, 1, 2])

        h, _,_ = ax[m, 1].hist(acs[:, 1], bins=np.arange(-3., 3. + binwidth, binwidth))
        ax[m, 1].fill_betweenx([0, h.max()], -1., 0., color='lightsalmon', alpha=0.1, label= 'Brake')
        ax[m, 1].fill_betweenx([0, h.max()],  0., 1., color='lightgreen', alpha=0.1, label='Throttle')
        h, _,_ = ax[m, 1].hist(acs[:, 1], bins=np.arange(-3., 3. + binwidth, binwidth), color='blue')
        ax[m, 1].set_xticks([-2, -1, 0 ,1, 2])

    ax[0, 0].legend()
    ax[0, 1].legend()

    ax[0, 0].set_ylabel('Gaussian')
    ax[1, 0].set_ylabel('Beta')

    plt.show()


def run_distr():
    #evaluate_CR('beta')
    #evaluate_CR('normal')
    pass

if __name__ == '__main__':
    #run_distr()
    #compare_CR()
    #compare_CR_acs()

    #find_observation()
    #sample_distribution()
    #sample_distribution2()

    evaluate_CR_nomerge('normal')