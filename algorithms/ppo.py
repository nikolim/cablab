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
env_name = "Assault-ram-v0"
env = gym.make(env_name)

n_state = 128 
n_actions = 7
episodes = 500
max_timesteps = 10000

memory = Memory()
ppo = PPO(n_state, n_actions)

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

illegal_pick_ups = []
illegal_moves = []
do_nothing = []
entropys = []
n_passengers = []
rewards = []

def normalize_array(ar): 
    divide_256 = lambda x: round((x/256),4)
    new_arr = map(divide_256, ar)
    return list(new_arr)

for episode in range(episodes):

    state = env.reset()
    state = normalize_array(state)
    saved_rewards = [0, 0, 0, 0]
    episode_reward = 0
    uncertainty = None
    n_steps = 0
    pick_ups = 0
    mean_entropy = 0

    number_of_action_4 = 0
    number_of_action_5 = 0
    wrong_pick_up_or_drop_off = 0

    while True:

        n_steps += 1
        action = ppo.policy_old.act(state, memory)

        state, reward, done, _ = env.step(action)
        state = normalize_array(state)
        saved_rewards = track_reward(reward, saved_rewards)

        episode_reward += reward
        memory.rewards.append(reward)
        memory.is_terminal.append(done)

        if done:
            mean_entropy = ppo.update(memory, episode)
            mean_entropy = round(mean_entropy, 3)
            memory.clear()
            print(f"Episode: {episode} Reward: {episode_reward}") 
            break
    
ppo.save_model(log_path)

