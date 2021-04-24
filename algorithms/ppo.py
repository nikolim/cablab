import os
import time
import json
import shutil
import numpy as np
from pyvirtualdisplay import Display

import gym
import gym_cabworld

from algorithms.ppo_models import *
from common.features import *
from common.logging import *
from common.plotting import *

cfg_path = "configs"
ppo_cfg_file = "ppo_conf.json"
ppo_cfg_file_path = os.path.join(cfg_path, ppo_cfg_file)

def train_ppo(n_episodes, version):

     # Start virtual display
    disp = Display()
    disp.start()

    # Load configuration
    cfg = json.load(open(ppo_cfg_file_path))

    env_name = 'Cabworld-' + version
    env = gym.make(env_name)

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    max_timesteps = env.spec.max_episode_steps

    log_path = create_log_folder("ppo")
    tracker = Tracker(version)

     # copy config-file into log-folder
    shutil.copyfile(ppo_cfg_file_path, os.path.join(log_path, ppo_cfg_file))

    memory = Memory()
    ppo = PPO(n_states, n_actions, cfg)

    # temp pick-up and drop-off counter for v1
    n_pick_ups = 0
    n_drop_offs = 0


    for episode in range(n_episodes):
        tracker.new_episode()
        state = env.reset()
        for _ in range(max_timesteps):
            action = ppo.policy_old.act(state, memory)
            old_state = state
            state, reward, done, _ = env.step(action)
            tracker.track_reward_and_action(reward, action, old_state)

             # additonal metric used to measure waiting time
            if version == 'v1':
                if reward == 1: 
                    if action == 4: 
                        n_pick_ups +=1 
                    else: 
                        n_drop_offs += 1

                if n_pick_ups == 2: 
                    tracker.reset_waiting_time(log=True)
                    n_pick_ups = 0
                if n_drop_offs == 2: 
                    tracker.reset_waiting_time(log=False)
                    n_drop_offs = 0

            memory.rewards.append(reward)
            memory.is_terminal.append(done)
            if done:
                if tracker.get_pick_ups() > cfg['min_pick_ups']:
                    ppo.update(memory, episode)
                memory.clear()
                print(
                    f"Episode: {episode} \t Reward: {round(tracker.episode_reward,3)} \t Passengers {tracker.get_pick_ups()}"
                )
                break

    ppo.save_model(log_path)
    tracker.plot(log_path)


def deploy_ppo(n_episodes, version, eval=False, render=False, wait=0.05):

    # load config
    current_folder = get_last_folder("ppo")
    cfg_file_path = os.path.join(current_folder, ppo_cfg_file)
    cfg = json.load(open(cfg_file_path))

    tracker = Tracker(version) if eval else False

    if not render:
        disp = Display()
        disp.start()

    env_name = "Cabworld-" + version
    env = gym.make(env_name)

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # load model
    ppo = PPO(n_state=n_states, n_actions=n_actions, cfg=cfg)
    current_model = os.path.join(current_folder, "ppo.pth")
    print(current_model)
    ppo.load_model(current_model)

    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = ppo.policy.deploy(state)
            old_state = state
            state, reward, done, _ = env.step(action)
            if eval:
                tracker.track_reward_and_action(reward, action, old_state)
            episode_reward += reward
            if render:
                env.render()
                time.sleep(float(wait))
            if done:
                print(
                    f'Episode: {episode} \t Reward: {round(episode_reward,3)}')
                break
    
    if eval:
        tracker.plot(log_path=current_folder, eval=True)
