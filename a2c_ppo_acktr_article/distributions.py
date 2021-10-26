import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr_article.utils import AddBias, init
import matplotlib.pyplot as plt


"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

#
# Standardize distribution interfaces
#

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


class FixedBeta(torch.distributions.Beta):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean



# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp()), action_mean, action_logstd.exp()


class DiagGaussUnitSTD(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussUnitSTD, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = torch.zeros(num_outputs).cuda()
        self.std = 1.                  # sigma = 0.5  --> sigma² = 0.25

    def forward(self, x):
        action_mean = self.fc_mean(x)
        action_logstd = self.logstd

        return FixedNormal(action_mean, action_logstd.exp()*self.std), action_mean, action_logstd.exp()



class DiagBeta(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagBeta, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.fc_alpha = init_(nn.Linear(num_inputs, num_outputs))
        self.fc_beta = init_(nn.Linear(num_inputs, num_outputs))
        self.sp = nn.Softplus()


    def forward(self, x):
        alpha = self.sp(self.fc_alpha(x)) + 1
        beta = self.sp(self.fc_beta(x)) + 1

        return FixedBeta(alpha, beta), alpha, beta



class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Bernoulli, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)

# CHECKS PPO
if __name__ == '__main__':

    input_th = torch.ones([8])
    input_th = input_th.unsqueeze(0)
    dist_shape = DiagGaussUnitSTD(8, 1)
    dist_curve = dist_shape(input_th)


    n = 1000
    sp_hist = []
    for i in range(n):
        s = dist_curve.sample()
        sp_hist.append(s.item() - dist_curve.mean.item())

    print(f'mean = {dist_curve.mean}')


    plt.hist(sp_hist, bins=100)
    plt.show()


