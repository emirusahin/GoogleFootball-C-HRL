#!/usr/bin/env python
# coding: utf-8

"""
TPU-only PPO training script for Google Football C-HRL, with per-environment reward tracking
and aggregated metrics. Launches via torch_xla multiprocessing, runs on v3-8/v4 TPUs.

Features:
1. Argument parsing for hyperparameters, TPU cores, GCS paths, and experiment options.
2. torch_xla integration (xla_multiprocessing) with SyncVectorEnv for stable multi-core training.
3. Refactored common model/env code into reusable functions.
4. Real-time per-step reward recording for each actor/env.
5. Saving of aggregated metrics plots per stage and per-env reward-vs-step plots (PNG) to GCS or local path.
6. Rank-based checkpointing: only process 0 writes files to avoid collisions.
7. Early stopping based on last N episodes and recording of steps per stage.
8. Episode-wise reward plots with smoothed moving average.
9. Final CSV summarizing steps taken per curriculum stage.
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib
matplotlib.use('Agg')  # headless backend
import matplotlib.pyplot as plt
from gym.vector import SyncVectorEnv
import gfootball.env as football_env
from gfootball.env.scenario_builder import all_scenarios
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp


def parse_args():
    parser = argparse.ArgumentParser(description="TPU PPO for Google Football C-HRL")
    parser.add_argument('--num_actors', type=int, default=16,
                        help='Number of parallel environments per process')
    parser.add_argument('--unroll_length', type=int, default=512,
                        help='Number of steps per PPO update')
    parser.add_argument('--train_epochs', type=int, default=4,
                        help='PPO epochs per update')
    parser.add_argument('--train_minibatches', type=int, default=8,
                        help='Minibatches per PPO update')
    parser.add_argument('--clip_range', type=float, default=0.08)
    parser.add_argument('--gamma', type=float, default=0.993)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--entropy_coef', type=float, default=0.003)
    parser.add_argument('--value_function_coef', type=float, default=0.5)
    parser.add_argument('--grad_norm_clip', type=float, default=0.64)
    parser.add_argument('--learning_rate', type=float, default=0.000343)
    parser.add_argument('--max_steps', type=int, default=5_000_000,
                        help='Max environment steps per curriculum stage')
    parser.add_argument('--early_stop_episodes', type=int, default=200,
                        help='Number of recent episodes to consider for early stopping')
    parser.add_argument('--early_stop_reward', type=float, default=1.9,
                        help='Average reward threshold for early stopping')
    parser.add_argument('--num_cores', type=int, default=8,
                        help='Number of TPU cores to use (must match xla_dist or xmp.spawn)')
    parser.add_argument('--save_path', type=str, default='gs://my-bucket/checkpoints/',
                        help='GCS path or local dir for checkpoints')
    parser.add_argument('--results_path', type=str, default='gs://my-bucket/results/',
                        help='GCS path or local dir for plots/metrics')
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
    device = xm.xla_device()
    is_master = (rank == 0)
    if is_master:
        os.makedirs(flags.save_path, exist_ok=True)
        os.makedirs(flags.results_path, exist_ok=True)

    curriculum = [
        'academy_empty_goal_close', 'academy_empty_goal', 'academy_run_to_score',
        'academy_run_to_score_with_keeper', 'academy_pass_and_shoot_with_keeper',
        'academy_run_pass_and_shoot_with_keeper', 'academy_3_vs_1_with_keeper',
        'academy_counterattack_easy', 'academy_counterattack_hard', 'academy_single_goal_versus_lazy',
        '1_vs_1_easy', '5_vs_5',
        '11_vs_11_easy_stochastic', '11_vs_11_stochastic', '11_vs_11_hard_stochastic',
    ]

    # track steps taken per curriculum stage
    stage_steps = {}

    # per-env reward tracking
    num_actors = flags.num_actors
    reward_hist = [[] for _ in range(num_actors)]
    step_hist = [[] for _ in range(num_actors)]

    for stage, env_name in enumerate(curriculum, start=1):
        # reset per-stage counters
        global_step = 0
        episode_rewards, episode_lengths, update_losses = [], [], []
        current_episode_rewards = [0.0] * num_actors
        current_episode_steps = [0] * num_actors

        # vector env
        envs = SyncVectorEnv([make_env(env_name) for _ in range(num_actors)])
        obs_np = envs.reset()
        obs = torch.from_numpy(obs_np).permute(0,3,1,2).float().to(device)

        # model & optimizer
        C, H, W = obs.shape[1:]
        model = ActorCritic((C,H,W), envs.single_action_space.n).to(device)
        optimizer = optim.Adam(model.parameters(), lr=flags.learning_rate)

        # resume from previous stage if available
        prev_ckpt = os.path.join(flags.save_path, f"ppo_{curriculum[stage-2]}.pth") if stage>1 else None
        if is_master and prev_ckpt and os.path.isfile(prev_ckpt):
            model.load_state_dict(torch.load(prev_ckpt, map_location=device))

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

                # record per-step and per-episode rewards
                for i in range(num_actors):
                    r = rews[i]
                    current_episode_rewards[i] += r
                    current_episode_steps[i] += 1
                    reward_hist[i].append(r)
                    step_hist[i].append(len(reward_hist[i]))
                    if dns[i]:
                        episode_rewards.append(current_episode_rewards[i])
                        episode_lengths.append(current_episode_steps[i])
                        current_episode_rewards[i] = 0.0
                        current_episode_steps[i] = 0

                # prepare for storage
                nxt = torch.from_numpy(nxt_obs).permute(0,3,1,2).float().to(device)
                ob_buf.append(obs)
                act_buf.append(acts)
                lp_buf.append(lp)
                val_buf.append(vals.squeeze())
                rew_buf.append(torch.from_numpy(rews).to(device))
                done_buf.append(torch.from_numpy(dns.astype(float)).to(device))
                obs = nxt
                global_step += num_actors

            # compute GAE & returns
            with torch.no_grad(): _, nv = model(obs); nv = nv.squeeze()
            ob_buf = torch.stack(ob_buf)
            act_buf = torch.stack(act_buf)
            lp_buf = torch.stack(lp_buf)
            val_buf = torch.stack(val_buf)
            rew_buf = torch.stack(rew_buf)
            done_buf = torch.stack(done_buf)
            adv, ret = compute_gae(rew_buf, val_buf, done_buf, nv, flags.gamma, flags.gae_lambda)

            # PPO update
            T, N = adv.shape
            batch_size = T * N
            idxs = np.arange(batch_size)
            for _ in range(flags.train_epochs):
                np.random.shuffle(idxs)
                mb = batch_size // flags.train_minibatches
                for start in range(0, batch_size, mb):
                    batch = idxs[start:start+mb]
                    b_obs = ob_buf.view(-1, C, H, W)[batch]
                    b_act = act_buf.view(-1)[batch]
                    b_lp0 = lp_buf.view(-1)[batch]
                    b_ret = ret.view(-1)[batch]
                    b_adv = adv.view(-1)[batch]

                    logits, vals = model(b_obs)
                    dist = Categorical(logits=logits)
                    lp1 = dist.log_prob(b_act)
                    ent = dist.entropy().mean()
                    ratio = (lp1 - b_lp0).exp()
                    pg = -torch.min(ratio * b_adv,
                                    torch.clamp(ratio, 1 - flags.clip_range, 1 + flags.clip_range) * b_adv).mean()
                    vloss = F.mse_loss(vals.squeeze(), b_ret)
                    loss = pg + flags.value_function_coef * vloss - flags.entropy_coef * ent

                    optimizer.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), flags.grad_norm_clip)
                    optimizer.step()
                    update_losses.append(loss.item())

            # early stopping check
            if is_master and len(episode_rewards) >= flags.early_stop_episodes:
                recent_avg = np.mean(episode_rewards[-flags.early_stop_episodes:])
                if recent_avg >= flags.early_stop_reward:
                    print(f"Early stopping at stage '{env_name}' after {len(episode_rewards)} episodes (avg={recent_avg:.2f})")
                    break

        # record steps for this stage
        stage_steps[env_name] = global_step

        if is_master:
            # save checkpoint
            ckpt_file = os.path.join(flags.save_path, f"ppo_{env_name}.pth")
            torch.save(model.state_dict(), ckpt_file)

            # plot episode rewards with smoothing
            if episode_rewards:
                episodes = np.arange(len(episode_rewards))
                plt.figure()
                plt.plot(episodes, episode_rewards, '.', ms=2, alpha=0.3, label='Episode Reward')
                window = 50
                if len(episode_rewards) >= window:
                    weights = np.ones(window) / window
                    smoothed = np.convolve(episode_rewards, weights, mode='valid')
                    plt.plot(np.arange(window-1, len(episode_rewards)), smoothed, '-', lw=2,
                             label=f'{window}-episode MA')
                plt.title(f"Episode Reward: {env_name}")
                plt.xlabel('Episode'); plt.ylabel('Reward')
                plt.legend()
                plt.savefig(os.path.join(flags.results_path, f'{env_name}_episode_reward.png'), dpi=300)
                plt.close()

    # after all stages: per-env and summary plots
    if is_master:
        # per-actor reward per step
        for i in range(num_actors):
            plt.figure()
            plt.plot(step_hist[i], reward_hist[i], marker='.', ms=2)
            plt.title(f"Env {i} Reward per Step")
            plt.xlabel('Step'); plt.ylabel('Reward')
            plt.savefig(os.path.join(flags.results_path, f'env_{i}_reward_step.png'), dpi=300)
            plt.close()

        # write stage steps summary
        summary_file = os.path.join(flags.results_path, 'stage_steps.csv')
        with open(summary_file, 'w') as f:
            f.write('env_name,steps\n')
            for env, steps in stage_steps.items():
                f.write(f"{env},{steps}\n")

    xm.rendezvous('end')


def main():
    flags = parse_args()
    xmp.spawn(train_loop, args=(flags,), nprocs=flags.num_cores, start_method='fork')


if __name__ == '__main__':
    main()
