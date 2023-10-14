import torch
import numpy as np
from a2c_ppo_acktr_article.action_scaling import scale_action
from a2c_ppo_acktr_article.arguments import get_args
from data_preprocessing import DataHandler
from cart_racing_v2 import CarRacing

class DemoGenerator():
    def __init__(self, device, args, env, actor_critic):
        self.device = device
        self.args = args
        self.env = env
        self.actor_critic = actor_critic
        self.eval_masks = torch.zeros(args.num_processes, 1, device=device)
        self.eval_recurrent_hidden_states = torch.zeros(
                                                    args.num_processes,
                                                    actor_critic.recurrent_hidden_state_size,
                                                    device=device)

    def run(self):
        counter = 0
        obs, _ = self.env.reset()
        while counter < 1000:
            counter += 1
            with torch.no_grad():
                obs = DataHandler().preprocess_images(
                    origin='to_ppo',
                    images_array=np.expand_dims(obs, axis=0))
                # obs = obs[:,6:90,6:90,:]
                _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                    torch.from_numpy(obs[0]).float().to(self.device).permute(2,0,1),
                    self.eval_recurrent_hidden_states,
                    self.eval_masks,
                    deterministic=True)
                
                action_env = scale_action(action, args)
                obs, rwd, done, infos = self.env.step(action_env)


if __name__ == '__main__':
    model_file = "CarRacing-v0_seed=2_nsteps_=500_d=beta_nup=1249.pt"
    device = 'cuda'
    args = get_args()
    env = CarRacing(render_mode="rgb-array")
    env.reset()
    actor_critic, obs_rms, _ = torch.load(model_file, map_location=device)
    demo_generator = DemoGenerator(device, args, env, actor_critic)
    demo_generator.run()