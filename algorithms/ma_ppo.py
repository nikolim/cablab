import os
import torch
import gym
import gym_cabworld
from torch.utils.tensorboard import SummaryWriter
from ma_ppo_models import Memory, ActorCritic, MAPPO
from tensorboard_tracker import track_reward, log_rewards

from tensorboard_tracker import log_rewards, track_reward, log_reward_uncertainty

from pyvirtualdisplay import Display

disp = Display().start()

torch.manual_seed(42)
env = gym.make("Cabworld-v7")

log_interval = 10
episodes = 3000
max_timesteps = 10000
update_timestep = 3000

memory = Memory()
mappo = MAPPO()

dirname = os.path.dirname(__file__)
log_path = os.path.join(dirname, "../runs", "ma_mappo")

if not os.path.exists(log_path):
    os.mkdir(log_path)
log_folders = os.listdir(log_path)
if len(log_folders) == 0:
    folder_number = 0
else:
    folder_number = max([int(elem) for elem in log_folders]) + 1

log_path = os.path.join(log_path, str(folder_number))
writer = SummaryWriter(log_path)

timestep = 0

for episode in range(episodes):

    states = env.reset()
    episode_reward = 0

    for t in range(max_timesteps):

        timestep += 1
        actions = mappo.policy_old.act(states, memory)
        states, rewards, done, _ = env.step([actions[0], actions[1]])

        episode_reward += sum(rewards)

        memory.rewards.append(rewards)
        memory.is_terminal.append(done)

        if timestep % update_timestep == 0:
            uncertainty = mappo.update(memory, episode)
            memory.clear()
            timestep = 0

        if done:
            print(f"Episode: {episode} Reward: {episode_reward}")
            break