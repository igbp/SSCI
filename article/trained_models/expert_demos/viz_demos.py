import numpy as np
import matplotlib.pyplot as plt

def viz_data():
    """Plots expert's demonstration observations and actions"""	
    obs_path = '/home/giovani/faire/dril/dril/demo_data/obs_CarRacing-v0_seed=66_ntraj=1.npy'
    obs = np.load(obs_path)
    acs_path = '/home/giovani/faire/dril/dril/demo_data/acs_CarRacing-v0_seed=66_ntraj=1.npy'
    acs = np.load(acs_path)

    fig, ax = plt.subplots(8, 8, figsize=(15, 15))
    start = 64
    for i in range(8 * 8):
        ax[i // 8, i % 8].imshow(obs[i+start, 0, :, :], cmap='gray')

    plt.show()


def plot_experts():
    """Plots expert actions for CarRacing-v0."""
    dir_path = '/home/giovani/hacer/dril/dril/demo_data'
    seeds = [68 , 66, 66]
    titles = ['PPO-Beta (Bounded)', 'PPO-Gaussian (Unbounded)', 'PPO-Gaussian (Clipped)']
    fig, ax = plt.subplots(2, len(seeds),figsize=(10, 5))
    actions = ['Left | Right', 'Brake | Throttle']
    for j, seed in enumerate(seeds):
        acs_file_name = f'acs_CarRacing-v0_seed={seed}_ntraj=1.npy'
        acs = np.load(os.path.join(dir_path, acs_file_name))
        for i in range(acs.shape[1]):
            act = np.clip(acs[:, i],-1,1) if j==2 else acs[:, i]
            ax[i, j].scatter(range(acs.shape[0]),act, s=1, color=f"{'blue' if i==0 else 'orange'}")
            ax[i, j].set_ylabel(f'{actions[i]}') if j==0 else None
            ax[i, j].set_ylim(-4.2, 4.2)
            ax[i, j].fill_between(range(acs.shape[0]), -1*np.ones(acs.shape[0]), 1*np.ones(acs.shape[0]),
                                  facecolor='gray', alpha=0.3, label='Action space')


            if i==0:
                ax[i, j].set_title(titles[j])
            else:
                ax[i, j].set_xlabel('Episode steps')

    ax[0,0].legend(loc='upper left')


    fig_name = "clipped_bounded_action_expert.jpg"
    save_path = os.path.join(dir_path, fig_name)
    print(f'saving figure to {save_path}')
    plt.savefig(save_path, format='jpg')
    plt.show()


