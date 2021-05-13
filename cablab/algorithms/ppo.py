import os
import time
import toml
import shutil
from pathlib import Path
import numpy as np
from pyvirtualdisplay import Display

import gym
import gym_cabworld

from cablab.algorithms.ppo_models import *
from cablab.common.features import *
from cablab.common.logging import *
from cablab.common.plotting import *

abs_path = Path(__file__).parent.absolute()
ppo_cfg_file = "ppo_conf.toml"
ppo_cfg_file_path = os.path.join(abs_path,"../configs",ppo_cfg_file)

def train_ppo(n_episodes, version):

     # Start virtual display
    disp = Display()
    disp.start()

    # Load configuration
    cfg = toml.load(open(ppo_cfg_file_path), _dict=dict)

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

    for episode in range(n_episodes):
        tracker.new_episode()
        state = env.reset()
        for _ in range(max_timesteps):
            action = ppo.policy_old.act(state, memory)
            old_state = state
            state, reward, done, _ = env.step(action)
            
            # reset action counter after spawn
            if version == 'v0':
                if passenger_spawn(old_state, state):
                    tracker.reset_action_counter()

            # additonal metric used to measure waiting time
            if version == 'v1':
                tracker.track_pick_up_time(reward, action)

            tracker.track_reward_and_action(reward, action, old_state)

            memory.rewards.append(reward)
            memory.is_terminal.append(done)
            if done:
                if tracker.get_pick_ups() >= cfg['min_pick_ups']:
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
    cfg = toml.load(open(cfg_file_path), _dict=dict)

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
