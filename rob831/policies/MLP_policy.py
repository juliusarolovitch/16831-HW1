import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from rob831.infrastructure import pytorch_util as ptu
from rob831.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers, size=self.size,
            )
            self.mean_net.to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        torch_obs = ptu.from_numpy(observation)

        action = self.forward(torch_obs).sample()
        
        return action.detach().cpu().numpy()

    def update(self, observations, actions, **kwargs):
        loss_f = kwargs['loss']
        
        obs_tensor = ptu.from_numpy(observations)
        actions_tensor = ptu.from_numpy(actions)

        predicted_dist = self.forward(obs_tensor)
        predicted_actions = predicted_dist.mean  

        loss = loss_f(predicted_actions, actions_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss


    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor) -> Any:
        
        if self.discrete:
            dist = torch.distributions.Categorical(logits=self.logits_na(observation))
        else:
            dist = torch.distributions.Normal(self.mean_net(observation), torch.exp(self.logstd))
        
        return dist

#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        # No fixed loss; compute NLL from distribution
        self.loss = None  # Will compute NLL directly

    def update(self, observations, actions, **kwargs):
        obs_tensor = ptu.from_numpy(observations)
        actions_tensor = ptu.from_numpy(actions)
        
        action_distribution = self.forward(obs_tensor)
        
        nll = -action_distribution.log_prob(actions_tensor).mean()
        
        self.optimizer.zero_grad()
        nll.backward()
        self.optimizer.step()

        return {'Training Loss': ptu.to_numpy(nll)}
