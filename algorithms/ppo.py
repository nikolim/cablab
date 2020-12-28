import os
import torch
import gym
import math
import gym_cabworld
from torch.utils.tensorboard import SummaryWriter
from ppo_models import Memory, ActorCritic, PPO
from tensorboard_tracker import track_reward, log_rewards
from plotting import *
from features import *

from pyvirtualdisplay import Display

disp = Display().start()

torch.manual_seed(42)
env_name = "Cabworld-v6"
env = gym.make(env_name)

log_interval = 10
episodes = 1000
max_timesteps = 10000
update_timestep = 10000

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
os.mkdir(log_path)
with open(os.path.join(log_path, "info.txt"), "w+") as info_file: 
    info_file.write(env_name + "\n")
    info_file.write("Episodes:" + str(episodes) + "\n")

timestep = 0

illegal_pick_ups = []
illegal_moves = []
do_nothing = []
entropys = []
n_passengers = []
rewards = []

for episode in range(episodes):

    state = env.reset()
    state = feature_engineering(state)
    saved_rewards = [0, 0, 0, 0]
    episode_reward = 0
    uncertainty = None
    n_steps = 0
    pick_ups = 0
    mean_entropy = 0

    for t in range(max_timesteps):

        timestep += 1
        n_steps += 1
        action = ppo.policy_old.act(state, memory)

        if action == 5: 
            saved_rewards[3] += 1

        state, reward, done, _ = env.step(action)
        state = feature_engineering(state)
        saved_rewards = track_reward(reward, saved_rewards)

        if reward == 100: 
            pick_ups += 1

        # increase reward -> increase passengers 
        reward = 1000 if reward == 100 else reward

        episode_reward += reward

        memory.rewards.append(reward)
        memory.is_terminal.append(done)

        if timestep % update_timestep == 0:
            mean_entropy = ppo.update(memory, episode)
            memory.clear()
            timestep = 0

        if done:
            print(f"Episode: {episode} Reward: {episode_reward} Steps {n_steps} Passengers {pick_ups//2} Entropy {mean_entropy}")
            break
    
    rewards.append(episode_reward)
    illegal_pick_ups.append(saved_rewards[1])
    illegal_moves.append(saved_rewards[2])
    do_nothing.append(saved_rewards[3])
    entropys.append(mean_entropy)
    n_passengers.append(pick_ups//2)

ppo.save_model(log_path)

from plotting import * 

plot_rewards(rewards, log_path)
plot_rewards_and_entropy(rewards, entropys, log_path)
plot_rewards_and_passengers(rewards, n_passengers, log_path)
plot_rewards_and_illegal_actions(rewards, illegal_pick_ups, illegal_moves, do_nothing,log_path)
