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
cfg_file = "dqn_conf.json"
cfg_file_path = os.path.join(cfg_path, cfg_file)

def train_dqn(n_episodes, env):

    # Start virtual display
    disp = Display()
    disp.start()

    # Load configuration
    cfg = json.load(open(cfg_file_path))
    n_hidden = cfg['n_hidden']
    lr = cfg['lr']
    update_freq = cfg['update_freq']
    gamma = cfg['gamma']
    epsilon = cfg['epsilon']
    epsilon_decay = cfg['epsilon_decay']
    epsilon_min = cfg['epsilon_min']
    replay_size = cfg['replay_size']
    target_update = cfg['target_update']
    munchhausen = cfg['munchhausen']
    assign_psng = cfg['assign_psng']
    min_pick_ups = cfg['min_pick_ups']
    episodes_without_training = cfg['episodes_without_training']

    env_name = 'Cabworld-' + env
    env = gym.make(env_name)

    n_states = env.observation_space.shape[0] + (1 if assign_psng else 0)
    n_actions = env.action_space.n

    log_path = create_log_folder("dqn")
    tracker = Tracker()

    # copy config-file into log-folder
    shutil.copyfile(cfg_file_path, os.path.join(log_path, cfg_file))

    dqn = DQN(n_states, n_actions, n_hidden, lr)
    memory = deque(maxlen=episodes_without_training *
                   env.spec.max_episode_steps)

    for episode in range(n_episodes + episodes_without_training):

        tracker.new_episode()
        if episode % target_update == 0:
            dqn.copy_target()

        policy = gen_epsilon_greedy_policy(dqn, epsilon, n_actions)
        state = env.reset()
        state = assign_passenger(state) if assign_psng else state
        is_done = False
        steps = 0

        while not is_done:
            steps += 1
            action = policy(state)
            next_state, reward, is_done, _ = env.step(action)
            tracker.track_reward_and_action(reward, action, state)

            # optional assign passengers to prepare for multi-agent-env
            if assign_psng:
                next_state, reward = single_agent_assignment(
                    reward, action, state, next_state, tracker)

            memory.append((state, action, next_state, reward, is_done))

            if episode > episodes_without_training and steps % update_freq == 0:
                if munchhausen:
                    dqn.replay_munchhausen(memory, replay_size, gamma)
                else:
                    dqn.replay(memory, replay_size, gamma)

            if is_done:
                print(
                    f"Episode: {episode} \t Reward: {round(tracker.episode_reward,3)} \t Passengers {tracker.get_pick_ups()}"
                )
                if tracker.get_pick_ups() < min_pick_ups:
                    for _ in range(min(env.spec.max_episode_steps, len(memory))):
                        memory.pop()
                break

            state = next_state

        if episode > episodes_without_training:
            epsilon = max(epsilon * epsilon_decay, epsilon_min)

    dqn.save_model(log_path)
    tracker.plot(log_path)


def deploy_dqn(n_episodes, env, eval=False, render=False, wait=0.05,):

    # load config
    current_folder = get_last_folder("dqn")
    cfg_file_path = os.path.join(current_folder, cfg_file)
    cfg = json.load(open(cfg_file_path))
    
    assign_psng = cfg['assign_psng']
    n_hidden = cfg['n_hidden']

    tracker = Tracker() if eval else False

    if not render:
        disp = Display()
        disp.start()

    env_name = "Cabworld-" + env
    env = gym.make(env_name)

    n_states = env.observation_space.shape[0] + (1 if assign_psng else 0)
    n_actions = env.action_space.n
    
    # load model
    dqn = DQN(n_states, n_actions, n_hidden)
    current_model = os.path.join(current_folder, "dqn.pth")
    print(current_model)
    dqn.load_model(current_model)

    for episode in range(n_episodes):
        if tracker:
            tracker.new_episode()
        state = env.reset()
        state = assign_passenger(state) if assign_psng else state
        episode_reward = 0
        done = False
        while not done:
            action = dqn.deploy(state)
            old_state = state
            state, reward, done, _ = env.step(action)
            if eval:
                tracker.track_reward_and_action(reward, action, old_state)
            if assign_passenger: 
                if reward == 1: 
                    if action == 4: 
                        state = tuple((list(state)) + [old_state[-1]])
                    else:
                        state = assign_passenger(state) if assign_psng else state
                else: 
                    state = assign_passenger(state) if assign_psng else state
            episode_reward += reward
            if render:
                env.render()
                time.sleep(wait)
            if done:
                print(f'Episode: {episode} \t Reward: {round(episode_reward,3)}')
                break
    if eval:
        tracker.plot(log_path=current_folder, eval=True)
