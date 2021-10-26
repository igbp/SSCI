import torch

# Scale action from 2D to 3D according to distribution

def scale_action(action, args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if args.env_name == 'CarRacing-v0':
        action_env = torch.zeros((args.num_processes, 3))
        if args.throttle_brake_merge:
            if args.distribution == 'beta':
                sc = torch.Tensor([[2., 2.]]).to(device)
                b = torch.Tensor([[-1., -1.]]).to(device)

                action_int = sc * action + b

                action_env[:, 0] = action_int[:, 0]
                action_env[:, 1] = torch.relu(action_int[:, 1])
                action_env[:, 2] = torch.relu(-action_int[:, 1])

            else:
                action_int = action
                action_env[:, 0] = action_int[:, 0]
                action_env[:, 1] = torch.relu(action_int[:, 1])
                action_env[:, 2] = torch.relu(-action_int[:, 1])
        else:
            # No merge needed
            action_env = action

    else:
        if args.distribution == 'beta':
            sc = torch.Tensor([[2., 2.]]).to(device)
            b = torch.Tensor([[-1., -1.]]).to(device)

            action_env = sc * action + b

        else:
            action_env = action

    return action_env