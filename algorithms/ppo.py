import os
import torch
import gym
import gym_cabworld
from torch.utils.tensorboard import SummaryWriter
from ppo_models import Memory, ActorCritic, PPO
from tensorboard_tracker import track_reward, log_rewards

from tensorboard_tracker import log_rewards, track_reward

from pyvirtualdisplay import Display

disp = Display().start()

torch.manual_seed(42)
env = gym.make("Cabworld-v5")

log_interval = 10
episodes = 3000
max_timesteps = 10000
update_timestep = 3000

memory = Memory()
ppo = PPO()

dirname = os.path.dirname(__file__)
log_path = os.path.join(dirname, "../runs", "ppo")
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

    state = env.reset()
    saved_rewards = (0, 0, 0)
    episode_reward = 0

    for t in range(max_timesteps):
        timestep += 1
        action = ppo.policy_old.act(state, memory)
        state, reward, done, _ = env.step(action)
        saved_rewards = track_reward(reward, saved_rewards)
        episode_reward += reward

        memory.rewards.append(reward)
        memory.is_terminal.append(done)

        if timestep % update_timestep == 0:
            ppo.update(memory, episode)
            memory.clear()
            timestep = 0

        if done:
            print(f"Episode: {episode} Reward: {episode_reward}")
            log_rewards(writer, saved_rewards, episode_reward, episode)
            break
