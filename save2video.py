import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import gym
import torch.nn.functional as F
from collections import deque
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out+residual)
        return out

class ActorCritic(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        C,H,W = input_shape
        self.num_actions = num_actions
        self.relu = nn.ReLU()

        # first group of layers
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1)
        self.max1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.res1a = ResidualBlock(16)
        self.res1b =ResidualBlock(16)

        # Second stage: 16 → 32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.max2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.res2a = ResidualBlock(32)
        self.res2b = ResidualBlock(32)

        # Third stage
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)
        self.max3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.res3a = ResidualBlock(32)
        self.res3b = ResidualBlock(32)

        # Fourth stage
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)
        self.max4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.res4a = ResidualBlock(32)
        self.res4b = ResidualBlock(32)

        # Dynamically compute FC input size
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            dummy = self.forward_conv(dummy)
            self.flattened_size = dummy.view(1, -1).size(1)

        self.fc = nn.Linear(self.flattened_size, 256)
        self.policy = nn.Linear(256, num_actions)
        self.value = nn.Linear(256, 1)

    def forward_conv(self, x):
        x = self.conv1(x)
        x = self.max1(x)
        x = self.res1a(x)
        x = self.res1b(x)

        x = self.conv2(x)
        x = self.max2(x)
        x = self.res2a(x)
        x = self.res2b(x)

        x = self.conv3(x)
        x = self.max3(x)
        x = self.res3a(x)
        x = self.res3b(x)

        x = self.conv4(x)
        x = self.max4(x)
        x = self.res4a(x)
        x = self.res4b(x)
        return x

    def forward(self,x):
        x = x /255
        x = self.forward_conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc(x))
        return self.policy(x), self.value(x)
#
#
# import os
# import torch
# import torch.nn.functional as F
# import gfootball.env as football_env
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# def load_policy(ckpt, obs_shape, num_actions):
#     H, W, C = obs_shape
#     net = ActorCritic((C, H, W), num_actions).to(device)
#     net.load_state_dict(torch.load(ckpt, map_location=device))
#     net.eval()
#     return net
#
# scenarios = {
#     'academy_empty_goal':                     'checkpoints/ppo_academy_empty_goal.pth',
#     'academy_empty_goal_close':               'checkpoints/ppo_academy_empty_goal_close.pth',
#     'academy_run_to_score':                   'checkpoints/ppo_academy_run_to_score.pth',
#     'academy_run_to_score_with_keeper':       'checkpoints/ppo_academy_run_to_score_with_keeper.pth',
#     'academy_pass_and_shoot_with_keeper':     'checkpoints/ppo_academy_pass_and_shoot_with_keeper.pth',
#     'academy_run_pass_and_shoot_with_keeper': 'checkpoints/ppo_academy_run_pass_and_shoot_with_keeper.pth',
#     'academy_3_vs_1_with_keeper':             'checkpoints/ppo_academy_3_vs_1_with_keeper.pth',
# }
#
# os.makedirs('videos', exist_ok=True)
#
# for scen, ckpt in scenarios.items():
#     print(f"▶ Running & dumping video for {scen}")
#
#     # 1) Create the output folder *before* making the env
#     outdir = os.path.join('videos', scen)
#     os.makedirs(outdir, exist_ok=True)
#
#     # 2) Headless env wrapped in PeriodicDumpWriter
#     #    - write_full_episode_dumps=True triggers the wrapper
#     #    - render=True tells the wrapper to actually call render() each dump
#     env = football_env.create_environment(
#         env_name=scen,
#         representation='extracted',
#         stacked=True,
#         number_of_left_players_agent_controls=1,
#         render=True,                   # used by the dump‐writer, not at creation
#         write_full_episode_dumps=True, # wrap for video dumping
#         dump_frequency=1,              # record *every* frame
#         logdir=outdir,                 # where to write episode_0.mp4
#         rewards='scoring,checkpoints'
#     )
#
#     # 3) Load your trained network
#     obs_shape   = env.observation_space.shape  # (H, W, C)
#     num_actions = env.action_space.n
#     policy      = load_policy(ckpt, obs_shape, num_actions)
#
#     # 4) Step through one episode – the wrapper will save `episode_0.mp4`
#     obs, done = env.reset(), False
#     while not done:
#         # permute HWC → CHW, batch, send to device
#         obs_t = (
#             torch.from_numpy(obs)
#             .permute(2, 0, 1)
#             .unsqueeze(0)
#             .float()
#             .to(device)
#         )
#         with torch.no_grad():
#             logits, _ = policy(obs_t)
#             probs     = F.softmax(logits, dim=-1)
#             action    = torch.distributions.Categorical(probs).sample().item()
#
#         obs, _, done, _ = env.step(action)
#
#     env.close()
#     print(f"✓ Video saved at {outdir}/episode_0.mp4\n")
#
# print("✅ All done! Each scenario’s MP4 is in `./videos/<scenario>/episode_0.mp4`.")
#
#
# import os
# import glob
# import subprocess
# import sys
#
# PY = sys.executable  # same Python that has gfootball installed
#
# def dump_to_avi(dump_path):
#     """Invoke GRF’s dump_to_video to turn a .dump → .avi"""
#     subprocess.run(
#         [PY, "-m", "gfootball.dump_to_video", "--trace_file", dump_path],
#         check=True
#     )
#     return dump_path.replace(".dump", ".avi")
#
# def avi_to_mp4(avi_path):
#     """Use ffmpeg to convert .avi → .mp4 (H.264)"""
#     mp4_path = avi_path.replace(".avi", ".mp4")
#     subprocess.run(
#         [
#             "ffmpeg", "-y", "-i", avi_path,
#             "-c:v", "libx264", "-pix_fmt", "yuv420p",
#             mp4_path
#         ],
#         check=True
#     )
#     return mp4_path
#
# def process_scenario_folder(folder):
#     dumps = sorted(glob.glob(os.path.join(folder, "*.dump")))
#     if not dumps:
#         print(f"  → no .dump files in {folder}")
#         return
#     for d in dumps:
#         print(f"  • Converting dump: {os.path.basename(d)}")
#         avi = dump_to_avi(d)
#         print(f"    → wrote AVI: {os.path.basename(avi)}")
#         mp4 = avi_to_mp4(avi)
#         print(f"    → wrote MP4: {os.path.basename(mp4)}")
#         # Optionally tidy up:
#         # os.remove(d)      # remove dump
#         # os.remove(avi)    # remove the intermediate AVI
#
# if __name__ == "__main__":
#     base_dir = "videos"
#     for scen in sorted(os.listdir(base_dir)):
#         scen_dir = os.path.join(base_dir, scen)
#         if os.path.isdir(scen_dir):
#             print(f"\n► Scenario: {scen}")
#             process_scenario_folder(scen_dir)
#     print("\n✅ All done! Look for .mp4 files alongside each .dump.")




# ## WORKS JUST ENDS EARLY
# import os
# import imageio
# import torch
# import torch.nn.functional as F
# import gfootball.env as football_env
#
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# def load_policy(ckpt, obs_shape, num_actions):
#     # obs_shape: (H, W, C)
#     H, W, C = obs_shape
#     net = ActorCritic((C, H, W), num_actions).to(device)
#     net.load_state_dict(torch.load(ckpt, map_location=device))
#     net.eval()
#     return net
#
# # Map each scenario to its checkpoint
# scenarios = {
#     'academy_empty_goal':                     'checkpoints/ppo_academy_empty_goal.pth',
#     'academy_empty_goal_close':               'checkpoints/ppo_academy_empty_goal_close.pth',
#     'academy_run_to_score':                   'checkpoints/ppo_academy_run_to_score.pth',
#     'academy_run_to_score_with_keeper':       'checkpoints/ppo_academy_run_to_score_with_keeper.pth',
#     'academy_pass_and_shoot_with_keeper':     'checkpoints/ppo_academy_pass_and_shoot_with_keeper.pth',
#     'academy_run_pass_and_shoot_with_keeper': 'checkpoints/ppo_academy_run_pass_and_shoot_with_keeper.pth',
#     'academy_3_vs_1_with_keeper':             'checkpoints/ppo_academy_3_vs_1_with_keeper.pth',
# }
#
# os.makedirs('videos', exist_ok=True)
#
# for scen, ckpt in scenarios.items():
#     print(f"\n▶ Recording scenario: {scen}")
#
#     # 1) Make env with rendering ON
#     env = football_env.create_environment(
#         env_name=scen,
#         representation='extracted',   # match your training repr
#         stacked=True,
#         number_of_left_players_agent_controls=1,
#         render=True,                  # open the SDL window
#         rewards='scoring,checkpoints'
#     )
#
#     # 2) Query shapes & load policy
#     obs_shape   = env.observation_space.shape  # (H, W, C)
#     num_actions = env.action_space.n
#     policy      = load_policy(ckpt, obs_shape, num_actions)
#
#     # 3) Setup video writer
#     video_path = os.path.join('videos', f'{scen}.mp4')
#     writer = imageio.get_writer(
#         video_path,
#         format='FFMPEG',
#         mode='I',
#         fps=10,
#         codec='libx264',
#         ffmpeg_params=['-pix_fmt', 'yuv420p']
#     )
#
#     # 4) Run one episode, capturing each frame
#     obs, done = env.reset(), False
#     while not done:
#         # grab the RGB frame from the window
#         frame = env.render(mode='rgb_array')
#         writer.append_data(frame)
#
#         # prepare obs tensor (H,W,C -> C,H,W -> batch)
#         obs_t = (
#             torch.from_numpy(obs)
#             .permute(2, 0, 1)   # (C,H,W)
#             .unsqueeze(0)       # (1,C,H,W)
#             .float()
#             .to(device)
#         )
#
#         with torch.no_grad():
#             logits, _ = policy(obs_t)         # [1, num_actions]
#             probs     = F.softmax(logits, dim=-1)
#             action    = torch.distributions.Categorical(probs).sample().item()
#
#         obs, _, done, _ = env.step(action)
#
#     # 5) Cleanup
#     writer.close()
#     env.close()
#     print(f"✓ Saved video: {video_path}")
#
# print("\n✅ All scenarios recorded. Check the `videos` folder for .mp4 files.")


import os
import imageio
import torch
import torch.nn.functional as F
import gfootball.env as football_env
# 0) Which device to run on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) Loader for your trained PPO
def load_policy(ckpt_path, obs_shape, num_actions):
    H, W, C = obs_shape           # e.g. (72, 96, 16)
    net = ActorCritic((C, H, W), num_actions).to(device)
    net.load_state_dict(torch.load(ckpt_path, map_location=device))
    net.eval()
    return net

# 2) Map each scenario to its checkpoint
scenarios = {
    'academy_empty_goal':                     'checkpoints/ppo_academy_empty_goal.pth',
    'academy_empty_goal_close':               'checkpoints/ppo_academy_empty_goal_close.pth',
    'academy_run_to_score':                   'checkpoints/ppo_academy_run_to_score.pth',
    'academy_run_to_score_with_keeper':       'checkpoints/ppo_academy_run_to_score_with_keeper.pth',
    'academy_pass_and_shoot_with_keeper':     'checkpoints/ppo_academy_pass_and_shoot_with_keeper.pth',
    'academy_run_pass_and_shoot_with_keeper': 'checkpoints/ppo_academy_run_pass_and_shoot_with_keeper.pth',
    'academy_3_vs_1_with_keeper':             'checkpoints/ppo_academy_3_vs_1_with_keeper.pth',
}

os.makedirs('../../../../Desktop/videos', exist_ok=True)

# 3) Loop & record
for scen, ckpt in scenarios.items():
    print(f"\n▶ Recording scenario: {scen}")

    # a) Create the env with extracted obs + live rendering
    env = football_env.create_environment(
        env_name=scen,
        representation='extracted',              # your trained representation
        stacked=True,
        number_of_left_players_agent_controls=1,
        render=True,                             # opens a 1280×720 window
        rewards='scoring,checkpoints'
    )

    # b) Load your policy
    obs_shape   = env.observation_space.shape   # (H, W, C)
    num_actions = env.action_space.n
    policy      = load_policy(ckpt, obs_shape, num_actions)

    # c) Set up a high‐quality, 30 FPS H.264 writer
    video_path = os.path.join('../../../../Desktop/videos', f'{scen}.mp4')
    writer = imageio.get_writer(
        video_path,
        format='FFMPEG',
        mode='I',
        fps=30,
        codec='libx264',
        ffmpeg_params=[
            '-pix_fmt', 'yuv420p',
            '-crf', '18',
            '-preset', 'slow'
        ]
    )

    # d) Run one full episode, capturing frames *after* each step
    obs = env.reset()
    # capture the very first kickoff frame
    writer.append_data(env.render(mode='rgb_array'))

    done = False
    while True:
        # prepare obs for the network: HWC → CHW → batch → device
        obs_t = (
            torch.from_numpy(obs)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(device)
        )

        # sample action
        with torch.no_grad():
            logits, _ = policy(obs_t)
            probs     = F.softmax(logits, dim=-1)
            action    = torch.distributions.Categorical(probs).sample().item()

        # step and then record the resulting frame
        obs, _, done, _ = env.step(action)
        writer.append_data(env.render(mode='rgb_array'))

        if done:
            break

    # e) Clean up
    writer.close()
    env.close()
    print(f"✓ Saved full‐HD MP4: {video_path}")

print("\n✅ Finished recording all scenarios. Enjoy your videos!")
