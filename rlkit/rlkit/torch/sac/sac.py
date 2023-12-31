from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.torch.core import np_to_pytorch_batch

from kmeans_pytorch import kmeans
import os
class SACTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            tdrp,
            vae,

            train_bipars=False,
            train_tdrp=False,
            train_vae = False,
            auxiliary_reward=False,
            tdrp_step=10,
            tdrp_pkl="../log/runs",

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.train_bipars = train_bipars

        self.tdrp = tdrp
        self.vae = vae

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )

        if train_tdrp:
            self.tdrp_criterion = nn.MSELoss()
            self.tdrp_optimizer = optimizer_class(
                self.tdrp.parameters(),
                lr = qf_lr,
            )
        if train_vae:
            self.vae_criterion = self.vae.loss_function
            self.vae_optimizer = optimizer_class(
                self.vae.parameters(),
                lr = qf_lr,
            )

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        self.train_tdrp=train_tdrp
        self.auxiliary_reward=auxiliary_reward
        self.tdrp_step = tdrp_step
        self.tdrp_pkl= tdrp_pkl

        self.pdist = torch.nn.PairwiseDistance(p=2)
    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Policy and Alpha Loss
        """
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs, reparameterize=True, return_log_prob=True,
        )
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1
        if self.train_bipars:
            q_value = torch.min(
                self.qf1(obs, new_obs_actions),
                self.qf2(obs, new_obs_actions),
            )
            new_next_actions, _, _, new_log_pi, *_ = self.policy(
                next_obs, reparameterize=True, return_log_prob=True,
            )
            q_next_value = torch.min(
                self.qf1(next_obs, new_next_actions),
                self.qf2(next_obs, new_next_actions),
            )
            f_value = q_value-self.discount*q_next_value
            q_new_actions = rewards+q_value*f_value

        else:
            q_new_actions = torch.min(
                self.qf1(obs, new_obs_actions),
                self.qf2(obs, new_obs_actions),
            )


        policy_loss = (alpha*log_pi - q_new_actions).mean()

        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        # Make sure policy accounts for squashing functions like tanh correctly!
        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs, reparameterize=True, return_log_prob=True,
        )
        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs, new_next_actions),
        ) - alpha * new_log_pi

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        """
        Update networks
        """
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.qf1, self.target_qf1, self.soft_target_tau
            )
            ptu.soft_update_from_to(
                self.qf2, self.target_qf2, self.soft_target_tau
            )

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()
        self._n_train_steps_total += 1

    def train_tdrp_from_torch(self, batch):
        terminals = batch['terminals']
        batch = np_to_pytorch_batch(batch)
        rewards = batch['rewards']

        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        obs = self.tdrp(obs)

        index = len(terminals) - self.tdrp_step-1
        count = 0
        loss = torch.zeros((1)).to(ptu.device)
        while index>=0:
            if terminals[index:index+self.tdrp_step].any():
                index -= 1
                continue
            loss += self.pdist(obs[index], obs[index+1:index+self.tdrp_step+1]).sum()
            index-=1

        index = len(terminals)-2*self.tdrp_step-1
        while index >= 0:
            distance = self.pdist(obs[index],obs[index+self.tdrp_step+1:index+2*self.tdrp_step+1])
            distance = 1-distance
            loss += torch.where(distance>torch.zeros_like(distance), distance, torch.zeros_like(distance).to(ptu.device)).sum()
            index -= 1
        tdrp_loss = self.tdrp_criterion(loss, torch.zeros_like(loss).to(ptu.device))
        self.tdrp_optimizer.zero_grad()
        tdrp_loss.backward()
        self.tdrp_optimizer.step()

        if self._need_to_update_eval_statistics:
            self.eval_statistics['tdrp Loss'] = loss.cpu().detach().numpy()

    def train_vae_from_torch(self, batch):
        batch = np_to_pytorch_batch(batch)

        obs = batch['observations']

        vae_loss = self.vae_criterion(obs)
        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
            self.tdrp,
            self.vae,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.qf1,
            target_qf2=self.qf2,
            tdrp=self.tdrp,
            vae=self.vae,
        )
