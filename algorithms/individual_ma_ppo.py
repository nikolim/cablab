import os, time
import torch
import gym
import gym_cabworld
from torch.utils.tensorboard import SummaryWriter
from ppo_models import Memory, ActorCritic, PPO
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
ppo = PPO()

memory2 = Memory()
ppo2 = PPO()

dirname = os.path.dirname(__file__)
log_path = os.path.join(dirname, "../runs", "ma_ppo")
log_path2 = os.path.join(dirname, "../runs", "ma_ppo2")

if not os.path.exists(log_path):
    os.mkdir(log_path)
log_folders = os.listdir(log_path)
if len(log_folders) == 0:
    folder_number = 0
else:
    folder_number = max([int(elem) for elem in log_folders]) + 1

log_path = os.path.join(log_path, str(folder_number))
log_path2 = os.path.join(log_path, str(folder_number))

writer = SummaryWriter(log_path)
writer2 = SummaryWriter(log_path2)

timestep = 0

for episode in range(episodes):

    state, state2 = env.reset()
    saved_rewards = (0, 0, 0)
    saved_rewards2 = (0, 0, 0)
    episode_reward = 0
    episode_reward2 = 0
    uncertainty = None
    uncertainty2 = None

    for t in range(max_timesteps):

        if episode >= 2995:
            env.render()
            time.sleep(0.05)

        timestep += 1
        action = ppo.policy_old.act(state, memory)
        action2 = ppo2.policy_old.act(state2, memory2)
        (state, state2), ma_rewards, done, _ = env.step([action, action2])

        saved_rewards = track_reward(ma_rewards[0], saved_rewards)
        saved_rewards2 = track_reward(ma_rewards[1], saved_rewards2)

        episode_reward += ma_rewards[0]
        episode_reward2 += ma_rewards[1]

        memory.rewards.append(ma_rewards[0])
        memory2.rewards.append(ma_rewards[1])
        memory.is_terminal.append(done)
        memory2.is_terminal.append(done)

        if timestep % update_timestep == 0:
            uncertainty = ppo.update(memory, episode)
            uncertainty2 = ppo2.update(memory2, episode)
            memory.clear()
            memory2.clear()
            timestep = 0

        if done:
            print(f"Episode: {episode} Rewards: {episode_reward} | {episode_reward2}")

            log_rewards(writer, saved_rewards, episode_reward, episode)
            log_rewards(writer2, saved_rewards2, episode_reward2, episode)

            if uncertainty:
                log_reward_uncertainty(writer, episode_reward, uncertainty, episode)
            if uncertainty2:
                log_reward_uncertainty(writer2, episode_reward2, uncertainty2, episode)
            break
