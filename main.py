import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr_article import algo, utils
from a2c_ppo_acktr_article.algo import gail
from a2c_ppo_acktr_article.algo.A2C_ACKTR import A2C_ACKTR
from a2c_ppo_acktr_article.algo.ppo import PPO
from a2c_ppo_acktr_article.arguments import get_args
from a2c_ppo_acktr_article.envs import make_vec_envs
from a2c_ppo_acktr_article.model_CR import Policy
from a2c_ppo_acktr_article.storage import RolloutStorage
from a2c_ppo_acktr_article.action_scaling import scale_action
from evaluation_article import evaluate


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    # Merging brake and throttle for CarRacing-v0
    if args.env_name == 'CarRacing-v0' and args.throttle_brake_merge:
        envs.action_space.shape = (2,)
        #input('\n\n\n We are here \n\n\n')

    training_score = {'step': [],
                      'avg_reward': []}


    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)
    """
    dir_path = '/home/giovani/article/trained_models/ppo/'
    file_path = 'Starting_CarRacing-v0_seed=2_nsteps_=500_d=normal_nup=1249.pt'

    actor_critic, _, _ = torch.load(dir_path + file_path, map_location='cuda:0')
    print(f'---- Loading pre-trainded model ----')
    print(f'{actor_critic}')
    print(f'------------------------------------')
    """

    if args.algo == 'a2c':
        agent = A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))

        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=20)
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)


    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    obs = envs.reset()

    rollouts.obs[0].copy_(obs)
    rollouts.to(device)


    for j in range(num_updates):
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            action_env = scale_action(action, args)
            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action_env)

            #--- RENDERS
            #envs.render()


            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])


            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()
        training_score['step'].append((j + 1) * args.num_processes * args.num_steps)
        training_score['avg_reward'].append(np.mean(episode_rewards))

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
            or j == num_updates-1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            file_name = args.env_name+ '_seed='+str(args.seed)+'_nsteps_='+str(args.num_steps)+"_d="+str(args.distribution)+ '_nup=' + str(j) + ".pt"

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'obs_rms', None),
                args
            ], os.path.join(save_path, file_name))

            path_scores =os.path.join(save_path, 'training_score_in' + str(len(episode_rewards))+'_episodes'+file_name)
            torch.save(training_score, path_scores)


        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()

            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                    .format(j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards), dist_entropy, value_loss,
                            action_loss))


        if (args.eval_interval is not None and len(episode_rewards) > args.eval_interval
            and j % args.eval_interval == 0):
            actor_critic, obs_rms = torch.load(file_name)

            evaluate(actor_critic, obs_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)

    envs.close()
    print(f'--- END ---')

if __name__ == "__main__":
    main()
