#!/usr/bin/env python
# coding: utf-8

"""
TPU-only PPO training script for Google Football C-HRL, with debug prints.
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gym.vector import SyncVectorEnv
import gfootball.env as football_env
from gfootball.env.scenario_builder import all_scenarios
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp


def parse_args():
    parser = argparse.ArgumentParser(description="TPU PPO for Google Football C-HRL (debug)")
    parser.add_argument('--num_actors', type=int, default=16)
    parser.add_argument('--unroll_length', type=int, default=512)
    parser.add_argument('--train_epochs', type=int, default=4)
    parser.add_argument('--train_minibatches', type=int, default=8)
    parser.add_argument('--clip_range', type=float, default=0.08)
    parser.add_argument('--gamma', type=float, default=0.993)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--entropy_coef', type=float, default=0.003)
    parser.add_argument('--value_function_coef', type=float, default=0.5)
    parser.add_argument('--grad_norm_clip', type=float, default=0.64)
    parser.add_argument('--learning_rate', type=float, default=0.000343)
    parser.add_argument('--max_steps', type=int, default=5_000_000)
    parser.add_argument('--early_stop_episodes', type=int, default=200)
    parser.add_argument('--early_stop_reward', type=float, default=1.9)
    parser.add_argument('--num_cores', type=int, default=8)
    parser.add_argument('--save_path', type=str, default='gs://my-bucket/checkpoints/')
    parser.add_argument('--results_path', type=str, default='gs://my-bucket/results/')
    return parser.parse_args()


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out + residual)
        return out


class ActorCritic(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        C, H, W = input_shape
        self.relu = nn.ReLU()
        # conv stages
        self.conv1 = nn.Conv2d(C, 16, 3)
        self.max1 = nn.MaxPool2d(3, 2)
        self.res1a = ResidualBlock(16)
        self.res1b = ResidualBlock(16)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.max2 = nn.MaxPool2d(3, 2)
        self.res2a = ResidualBlock(32)
        self.res2b = ResidualBlock(32)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.max3 = nn.MaxPool2d(3, 2)
        self.res3a = ResidualBlock(32)
        self.res3b = ResidualBlock(32)
        self.conv4 = nn.Conv2d(32, 32, 3)
        self.max4 = nn.MaxPool2d(3, 2)
        self.res4a = ResidualBlock(32)
        self.res4b = ResidualBlock(32)
        # dynamic flatten
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            tmp = self._forward_conv(dummy)
            flattened = tmp.view(1, -1).size(1)
        self.fc = nn.Linear(flattened, 256)
        self.policy = nn.Linear(256, num_actions)
        self.value = nn.Linear(256, 1)

    def _forward_conv(self, x):
        x = self.conv1(x); x = self.max1(x)
        x = self.res1a(x); x = self.res1b(x)
        x = self.conv2(x); x = self.max2(x)
        x = self.res2a(x); x = self.res2b(x)
        x = self.conv3(x); x = self.max3(x)
        x = self.res3a(x); x = self.res3b(x)
        x = self.conv4(x); x = self.max4(x)
        x = self.res4a(x); x = self.res4b(x)
        return x

    def forward(self, x):
        x = x / 255.0
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc(x))
        return self.policy(x), self.value(x)


def make_env(name):
    def thunk():
        return football_env.create_environment(
            env_name=name,
            representation='extracted',
            stacked=True,
            number_of_left_players_agent_controls=1,
            render=False,
            rewards='scoring,checkpoints'
        )
    return thunk


def compute_gae(rews, vals, dones, next_val, gamma, lam):
    T, N = rews.shape
    adv = torch.zeros_like(rews)
    last = torch.zeros(N, device=rews.device)
    for t in reversed(range(T)):
        mask = 1 - dones[t]
        delta = rews[t] + gamma * next_val * mask - vals[t]
        adv[t] = last = delta + gamma * lam * mask * last
        next_val = vals[t]
    return adv, adv + vals


def train_loop(rank, flags):
    print(f"[DEBUG] train_loop start rank={rank} @ {time.strftime('%X')}")
    device = xm.xla_device()
    print(f"[DEBUG] rank={rank} on device {device}")
    is_master = (rank == 0)
    if is_master:
        os.makedirs(flags.save_path, exist_ok=True)
        os.makedirs(flags.results_path, exist_ok=True)

    curriculum = [
        'academy_empty_goal_close', 'academy_empty_goal', 'academy_run_to_score',
        'academy_run_to_score_with_keeper', 'academy_pass_and_shoot_with_keeper',
        'academy_run_pass_and_shoot_with_keeper', 'academy_3_vs_1_with_keeper',
        'academy_counterattack_easy', 'academy_counterattack_hard',
        'academy_single_goal_versus_lazy', '1_vs_1_easy', '5_vs_5',
        '11_vs_11_easy_stochastic', '11_vs_11_stochastic', '11_vs_11_hard_stochastic',
    ]
    stage_steps = {}

    num_actors = flags.num_actors
    reward_hist = [[] for _ in range(num_actors)]
    step_hist = [[] for _ in range(num_actors)]

    for env_name in curriculum:
        print(f"[DEBUG] rank={rank} building SyncVectorEnv for '{env_name}'")
        envs = SyncVectorEnv([make_env(env_name) for _ in range(num_actors)])
        print(f"[DEBUG] rank={rank} created envs, now resetting")
        obs_np = envs.reset()
        print(f"[DEBUG] rank={rank} reset returned shape {obs_np.shape}")
        obs = torch.from_numpy(obs_np).permute(0,3,1,2).float().to(device)

        model = ActorCritic((obs.size(1),obs.size(2),obs.size(3)), envs.single_action_space.n).to(device)
        print(f"[DEBUG] rank={rank} built ActorCritic model")
        optimizer = optim.Adam(model.parameters(), lr=flags.learning_rate)

        global_step = 0

        # main training loop
        while global_step < flags.max_steps:
            ob_buf, act_buf, lp_buf, val_buf = [], [], [], []
            rew_buf, done_buf = [], []

            # rollout
            for _ in range(flags.unroll_length):
                logits, vals = model(obs)
                dist = Categorical(logits=logits)
                acts = dist.sample()
                lp = dist.log_prob(acts)
                nxt_obs, rews, dns, _ = envs.step(acts.cpu().numpy())

                for i in range(num_actors):
                    r = rews[i]
                    # ... reward tracking omitted for brevity ...

                nxt = torch.from_numpy(nxt_obs).permute(0,3,1,2).float().to(device)
                ob_buf.append(obs); act_buf.append(acts)
                lp_buf.append(lp); val_buf.append(vals.squeeze())
                rew_buf.append(torch.from_numpy(rews).to(device))
                done_buf.append(torch.from_numpy(dns.astype(float)).to(device))
                obs = nxt
                global_step += num_actors

                if is_master and global_step % 100 == 0:
                    print(f"[DEBUG] Stage '{env_name}' | Env steps: {global_step}")

            print(f"[DEBUG] rank={rank} completed one rollout, breaking for debug")
            break  # remove for full run

        envs.close()
        stage_steps[env_name] = global_step
        print(f"[DEBUG] rank={rank} finished '{env_name}' at step {global_step}")

    if is_master:
        print(f"[DEBUG] Master rank writing results, stages: {stage_steps}")

    xm.rendezvous('end')
    print(f"[DEBUG] rank={rank} rendezvous exit")


def main():
    flags = parse_args()
    print(f"[DEBUG] main starting with flags: {flags}")
    if flags.num_cores == 1:
        print("[DEBUG] Single-core mode")
        train_loop(0, flags)
    else:
        print(f"[DEBUG] spawning {flags.num_cores} TPU processes")
        xmp.spawn(train_loop, args=(flags,), nprocs=None, start_method='fork')


if __name__ == '__main__':
    main()
