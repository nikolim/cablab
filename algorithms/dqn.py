import os
import json
import time
import shutil
import logging
import numpy as np
from collections import deque
from pyvirtualdisplay import Display

import gym
import gym_cabworld

from algorithms.dqn_model import *
from common.logging import *
from common.features import *

cfg_path = "configs"
dqn_cfg_file = "dqn_conf.json"
dqn_cfg_file_path = os.path.join(cfg_path, dqn_cfg_file)

def train_dqn(n_episodes, version):

    # Start virtual display
    disp = Display()
    disp.start()

    # Load configuration
    cfg = json.load(open(dqn_cfg_file_path))

    env_name = 'Cabworld-' + version
    env = gym.make(env_name)

    n_states = env.observation_space.shape[0] + (1 if cfg['assign_psng'] else 0)
    n_actions = env.action_space.n

    log_path = create_log_folder("dqn")
    tracker = Tracker(version)

    # copy config-file into log-folder
    shutil.copyfile(dqn_cfg_file_path, os.path.join(log_path, dqn_cfg_file))

    dqn = DQN(n_states, n_actions, cfg)
    memory = deque(maxlen=cfg['episodes_without_training'] *
                   env.spec.max_episode_steps)

    for episode in range(n_episodes + cfg['episodes_without_training']):

        tracker.new_episode()

        if episode % cfg['target_update'] == 0:
            dqn.copy_target()

        # temp pick-up and drop-off counter for v1
        n_pick_ups = 0
        n_drop_offs = 0

        policy = gen_epsilon_greedy_policy(dqn, cfg['epsilon'], n_actions)
        state = env.reset()
        state = assign_passenger(state) if cfg['assign_psng'] else state
        is_done = False
        steps = 0

        while not is_done:
            steps += 1
            action = policy(state)
            next_state, reward, is_done, _ = env.step(action)
            tracker.track_reward_and_action(reward, action, state)

            # additonal metric used to measure waiting time
            if version == 'v1':
                if reward == 1: 
                    if action == 4: 
                        n_pick_ups +=1
                        tracker.add_waiting_time() 
                    else: 
                        n_drop_offs += 1
                if n_pick_ups == 2: 
                    tracker.reset_waiting_time()
                    n_pick_ups = 0
                if n_drop_offs == 2: 
                    tracker.reset_waiting_time()
                    n_drop_offs = 0
                    
            # optional assign passengers to prepare for multi-agent-env
            if cfg['assign_psng']:
                next_state, reward = single_agent_assignment(
                    reward, action, state, next_state, tracker)

            memory.append((state, action, next_state, reward, is_done))

            if episode > cfg['episodes_without_training'] and steps % cfg['update_freq'] == 0:
                if cfg['munchhausen']:
                    dqn.replay_munchhausen(memory, cfg['replay_size'], cfg['gamma'])
                else:
                    dqn.replay(memory, cfg['replay_size'], cfg['gamma'])

            if is_done:
                print(
                    f"Episode: {episode} \t Reward: {tracker.episode_reward:.2f} \t Passengers {tracker.get_pick_ups()}"
                )
                if tracker.get_pick_ups() < cfg['min_pick_ups']:
                    for _ in range(min(env.spec.max_episode_steps, len(memory))):
                        memory.pop()
                break

            state = next_state

        if episode > cfg['episodes_without_training']:
            cfg['epsilon'] = max(cfg['epsilon'] * cfg['epsilon_decay'], cfg['epsilon_min'])

    dqn.save_model(log_path)
    tracker.plot(log_path, )


def deploy_dqn(n_episodes, version, eval=False, render=False, wait=0.05):

    # load config
    current_folder = get_last_folder("dqn")
    cfg_file_path = os.path.join(current_folder, dqn_cfg_file)
    cfg = json.load(open(cfg_file_path))
    print(f'Config loaded: {cfg_file_path}')

    tracker = Tracker(version) if eval else False

    if not render:
        disp = Display()
        disp.start()

    env_name = "Cabworld-" + version
    env = gym.make(env_name)

    n_states = env.observation_space.shape[0] + (1 if cfg['assign_psng'] else 0)
    n_actions = env.action_space.n
    
    # load model
    dqn = DQN(n_states, n_actions, cfg)
    current_model = os.path.join(current_folder, "dqn.pth")
    print(current_model)
    dqn.load_model(current_model)

    for episode in range(n_episodes):
        if tracker:
            tracker.new_episode()
        state = env.reset()
        state = assign_passenger(state) if cfg['assign_psng'] else state
        episode_reward = 0
        done = False
        while not done:
            action = dqn.deploy(state)
            old_state = state
            state, reward, done, _ = env.step(action)
            if eval:
                tracker.track_reward_and_action(reward, action, old_state)
            if cfg['assign_psng']: 
                state, _ = single_agent_assignment(
                    reward, action, old_state, state, tracker)                
            episode_reward += reward
            if render:
                env.render()
                time.sleep(wait)
            if done:
                print(f'Episode: {episode} \t Reward: {round(episode_reward,3)}')
                break
    if eval:
        tracker.plot(log_path=current_folder, eval=True)
