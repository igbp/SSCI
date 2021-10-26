import matplotlib.pyplot as plt
import numpy as np
import torch

from a2c_ppo_acktr_article import utils
from a2c_ppo_acktr_article.envs import make_vec_envs
from a2c_ppo_acktr_article.action_scaling import scale_action
from a2c_ppo_acktr_article.arguments import get_args


def evaluate_loadedm_std():
    args = get_args()
    env_name = args.env_name
    num_processes = 1
    device = 'cuda:0'
    deterministic = True
    seed = 1
    eval_log_dir = '/home/giovani/article/trained_models/ppo/'
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True)

    dir_path = '/home/giovani/article/trained_models/ppo/'
    file_name = 'LunarLanderContinuous-v2_nsteps_=128_normal_nup=975.pt'

    path = dir_path + file_name
    actor_critic, obs_rms, args = torch.load(path)


    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    rollout = {'obs': [],
               'acs': [],
               'rwd': [],
               'scr': []}

    step = 0
    samples = []

    while len(eval_episode_rewards) < 1:

        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=deterministic)

        verbose = True
        if (step==50 or step==100 or step==150) and verbose:

            n_samples = 1000
            acs = []

            for i in range(n_samples):
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs,
                        eval_recurrent_hidden_states,
                        eval_masks,
                        deterministic=False)

                acs.append(action)
            samples.append(torch.cat(acs))

        # Scale action from 2D to 3D according to distribution
        action_env = scale_action(action, args)

        # Obser reward and next obs
        rollout['obs'].append(obs)

        obs, rwd, done, infos = eval_envs.step(action_env)
        rollout['rwd'].append(rwd)
        rollout['acs'].append(action_env)

        eval_envs.render()
        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])
                rollout['scr'].append(info['episode']['r'])

        step += 1

    fig, ax = plt.subplots(len(samples), 2, figsize=(12, 4), sharex=True, sharey=True)
    fig.suptitle(f'Deterministic={deterministic}, score={eval_episode_rewards}')
    binwidth = 0.01
    for i, acs in enumerate(samples):
        acs = acs.T.cpu().numpy()
        print(acs.shape)
        if args.env_name=='CarRacing-v0':
            ax[i, 0].hist(acs[0, :], bins=np.arange(-1., 1. + binwidth, binwidth))
            ax[i, 1].hist(acs[1, :], bins=np.arange(-1., 1. + binwidth, binwidth))
        else:
            ax[i, 0].hist(acs[0, :], bins=100)
            ax[i, 1].hist(acs[1, :], bins=100)


    plt.show()

    eval_envs.close()

    print('------ Eval Rewards ------')
    print(eval_episode_rewards)
    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))

    roll_name = file_name + '_rollout.pt'
    roll_path = dir_path + roll_name
    torch.save(rollout, roll_path)


def viz_rollout():

    dir_path = '/home/giovani/article/trained_models/ppo/20210718_CR_beta_scale=1_nstep=500/'
    file_name = 'CarRacing-v0_beta_sc=+1.0_nupdates=2499.pt'

    roll_name = file_name + '_rollout.pt'
    roll_path = dir_path + roll_name

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

    ax[2].plot(acs[:, 2])
    ax[2].set_ylabel('Action[2]')
    ax[2].plot(t * 1)
    ax[2].plot(t * -1)
    ax[2].set_xlabel('Time step')


    plt.show()


def evaluate_loadedm_performance_LLC():
    device = 'cuda:0'


    eval_log_dir = '/home/giovani/article/trained_models/ppo/'

    dir_path = '/home/giovani/article/trained_models/ppo_LunarLander_final/'
    file_name = 'LunarLanderContinuous-v2_seed=4_nsteps_=2048_d=beta_nup=487.pt'
    seed = 10
    print(f'seed={seed}')

    deterministic=False

    path = dir_path + file_name
    actor_critic, obs_rms, args = torch.load(path)
    args.num_processes = 1

    eval_envs = make_vec_envs(args.env_name, seed + args.num_processes, args.num_processes,
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
    thresh = 900 if args.env_name == 'CarRacing-v0' else 200
    print(f'... Starting simulation ...')
    n_episodes = 100
    while len(eval_episode_rewards) < n_episodes:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=deterministic)

        rollout['obs'].append(obs)
        rollout['acs'].append(action)

        verbose = False
        if (step==50 or step==100 or step==150) and verbose:

            n_samples = 1000
            acs = []
            for i in range(n_samples):
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs,
                        eval_recurrent_hidden_states,
                        eval_masks,
                        deterministic=False)
                acs.append(action)
            acs = torch.cat(acs).cpu().numpy()
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            fig.suptitle(f'{n_samples} of pi @ step = {step}', fontsize=12)
            ax[0].hist(acs[:, 0], bins=100)
            ax[1].hist(acs[:, 1], bins=100)
            plt.show()

        # action scaling
        # Scale action from 2D to 3D according to distribution
        action_env = scale_action(action, args)

        # Obser reward and next obs

        obs, rwd, done, infos = eval_envs.step(action_env)

        rollout['rwd'].append(rwd)

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
                print(f'Episode length = {info["episode"]["l"]}')
                if info['episode']['r'] < thresh:
                    fail_rate+=1
        step += 1


    eval_envs.close()

    print('------ Eval Rewards ------')
    print(eval_episode_rewards)
    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))

    file_test = 'test_score_nep='+str(n_episodes)+'_'+file_name
    print(f'\n... Saving file to \n{dir_path} \n{file_test}\n')
    torch.save(eval_episode_rewards, dir_path + file_test)


def plot_evaluation_LLC():
    dir_path = '/home/giovani/article/trained_models/ppo/'
    file_name = 'LunarLanderContinuous-v2_seed=4_nsteps_=2048_d=beta_nup=487.pt'

    test_file_name = 'test_score_nep=100_LunarLanderContinuous-v2_seed=4_nsteps_=2048_d=beta_nup=487.pt'
    thresh = 200.

    eval_episode_rewards = torch.load(dir_path + test_file_name)
    _, _, args = torch.load(dir_path+file_name)
    fail_rate=0
    for i in range(len(eval_episode_rewards)):
        if eval_episode_rewards[i] < thresh:
            fail_rate+=1

    threshold = np.ones_like(eval_episode_rewards)*thresh
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.suptitle(f'Fail rate {fail_rate/ len(eval_episode_rewards)*100}% '
                 f'nsteps={args.num_steps}'
                 
                 f'mean_score={np.mean(eval_episode_rewards):.2f}  +/- {np.std(eval_episode_rewards):.2f} \n {file_name}')
    ax.plot(eval_episode_rewards, linestyle='', marker='o')
    ax.plot(threshold)
    ax.set_ylim([-100, thresh+150])
    plt.show()


def evaluate_loadedm_performance_CR():
    device = 'cuda:0'

    eval_log_dir = '/home/giovani/article/trained_models/ppo/'

    idx = 1

    dir_path = '/home/giovani/article/trained_models/ppo/'
    file_name = 'CarRacing-v0_seed='+str(idx)+'_nsteps_=500_d=beta_nup=1249.pt'

    seed = idx*10 +20
    print(f'seed={seed} HERE')

    deterministic = True
    path = dir_path + file_name
    actor_critic, obs_rms, args = torch.load(path, map_location=device)
    args.num_processes = 20

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
    thresh = 900 if args.env_name == 'CarRacing-v0' else 200
    print(f'... Starting simulation ...')
    n_episodes = 100
    while len(eval_episode_rewards) < n_episodes:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=deterministic)

        rollout['obs'].append(obs)
        rollout['acs'].append(action)

        verbose = False
        if (step == 50 or step == 100 or step == 150) and verbose:

            n_samples = 1000
            acs = []
            for i in range(n_samples):
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs,
                        eval_recurrent_hidden_states,
                        eval_masks,
                        deterministic=False)
                acs.append(action)
            acs = torch.cat(acs).cpu().numpy()
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            fig.suptitle(f'{n_samples} of pi @ step = {step}', fontsize=12)
            ax[0].hist(acs[:, 0], bins=100)
            ax[1].hist(acs[:, 1], bins=100)
            plt.show()

        # action scaling
        # Scale action from 2D to 3D according to distribution
        action_env = scale_action(action, args)

        # Random agent
        #action_env = torch.Tensor([eval_envs.action_space.sample()])

        # Obser reward and next obs

        obs, rwd, done, infos = eval_envs.step(action_env)


        #eval_envs.render()

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
                print(f'Episode length = {info["episode"]["l"]}')
                if info['episode']['r'] < thresh:
                    fail_rate += 1
        step += 1

    eval_envs.close()

    print('------ Eval Rewards ------')
    print(eval_episode_rewards)
    print(" Evaluation using {} episodes: \n mean reward {:.5f}  std: {:.5f}\n ".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards[0:100]),  np.std(eval_episode_rewards[0:100])))

    file_test = 'test_score_nep=' + str(n_episodes) + '_' + file_name
    rollout_file = 'rollout_'+file_name
    print(f'\n... Saving file to \n{dir_path} \n{file_test}\n')
    torch.save(eval_episode_rewards, dir_path + file_test)
    torch.save(rollout, dir_path + rollout_file)

    return file_name, rollout_file, dir_path


def plot_evaluation_CR(file_name: str):
    dir_path = '/home/giovani/article/trained_models/ppo/'
    #file_name = 'CarRacing-v0_seed=1_nsteps_=500_d=beta_nup=1249.pt'

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


def score_check():
    dir_path = '/home/giovani/article/trained_models/ppo/'
    file_name = 'LunarLanderContinuous-v2_nsteps_=128_normal_nup=975.pt'
    _, _, args = torch.load(dir_path+ file_name)
    score_path = 'training_score_in10_episodes' + file_name

    tscore = torch.load(dir_path + score_path)

    fig, ax = plt.subplots(2,1, figsize=(12, 8))
    fig.suptitle(f'beta_scale={args.beta_scale} \n {file_name}')
    #ax.plot(tscore['stp'], tscore['med'], marker='o', linestyle=' ')
    ax[0].plot(tscore['stp'], tscore['avg'])
    ax[0].set_ylabel(f'Avg')

    ax[1].plot(tscore['stp'], tscore['max'])
    ax[1].plot(tscore['stp'], tscore['min'])
    ax[1].set_ylabel(f'Max/Min')

    plt.show()


def ep_track():
    dir_path = '/home/giovani/article/trained_models/ppo/'
    files = ['CarRacing-v0_nsteps_=1000_d=beta_nup=999.pt',
             'CarRacing-v0_nsteps_=1000_d=beta_nup=999.pt']

    fig, ax = plt.subplots(1, 3, figsize=(20, 12), sharey=True)


    avgs = []
    for i, file_name in enumerate(files):
        _, _, args = torch.load(dir_path + file_name)

        scr_path = 'training_score_in10_episodes'+ file_name

        training_score = torch.load(dir_path+scr_path)
        avg = []
        mov_window = 200
        t = np.linspace(0, len(training_score['reward'])-1, len(training_score['reward']))
        t_avg = np.linspace(mov_window, len(training_score['reward'])-1, len(training_score['reward'])-mov_window)

        for j in range(len(training_score['reward'])-mov_window):
            avg.append(np.mean(training_score['reward'][j-mov_window:j]))
        avgs.append(avg)

        ax[i].plot(t, training_score['reward'])
        ax[i].plot(t_avg, avg)
        ax[i].set_title(file_name)
        ax[i].grid()



    ax[2].plot(avgs[0])
    ax[2].plot(avgs[1])
    ax[2].grid()

    plt.show()


def rollout_comparison():

    file_name = 'CarRacing-v0_seed=1_nsteps_=500_d=beta_nup=1249.pt'
    rollout_file = 'rollout_' + file_name
    dir_path = '/home/giovani/article/trained_models/ppo/'

    rollout = torch.load(dir_path + rollout_file)
    #actor_critic, obs_rms, args = torch.load(dir_path + file_name, map_location='cpu')

    """
    fig, ax = plt.subplots(3, 1, figsize=(12, 8))

    acs = torch.cat(rollout['acs'])*torch.Tensor([2]).to('cuda:0') + torch.Tensor([-1, -1]).to('cuda:0')
    acs = acs.cpu().numpy()

    ax[0].plot(acs[:, 0])
    ax[1].plot(torch.relu(torch.Tensor(acs[:, 1])).numpy())
    ax[2].plot(torch.relu(torch.Tensor(-acs[:, 1])).numpy())

    plt.show()
    """

    fig, ax = plt.subplots()
    rwd = torch.cat(rollout['rwd']).numpy()
    print(rwd)


def check_params():
    dir = '/home/giovani/article/trained_models/ppo_article/'
    file_name = 'CarRacing-v0_seed=1_nsteps_=500_d=normal_nup=1249.pt'

    _,_,p = torch.load(dir+file_name)
    print(p)

if __name__ == '__main__':
    #check_params()

    #file_name, rollout_file, dir_path = evaluate_loadedm_performance_CR()
    #plot_evaluation_CR(file_name)

    rollout_comparison()

    #evaluate_loadedm_performance_LLC()
    #evaluation_100ep()
    #ep_track()
    #evaluate_loadedm_std()
    #viz_rollout()
    #score_check()