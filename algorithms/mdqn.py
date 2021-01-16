import os
import time
import torch
import numpy as np
from pyvirtualdisplay import Display

import gym
import gym_cabworld

from algorithms.m_dqn_model import M_DQN_Agent
from common.features import clip_state, cut_off_state
from common.logging import create_log_folder, get_last_folder
from common.logging import Tracker


def train_mdqn(n_episodes):

    Display().start()
    env_name = "Cabworld-v0"
    env = gym.make(env_name)

    n_states = 14
    n_actions = 6
    max_timesteps = 1000

    log_path = create_log_folder("mdqn")
    tracker = Tracker()

    layer_size = 64
    buffer_size = 50000
    batch_size = 64
    eps_decay = 0.95
    eps = 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mdqn = M_DQN_Agent(
        state_size=n_states,
        action_size=n_actions,
        layer_size=layer_size,
        BATCH_SIZE=batch_size,
        BUFFER_SIZE=buffer_size,
        LR=0.001,
        TAU=0.01,
        GAMMA=0.99,
        UPDATE_EVERY=10,
        device=device,
        seed=42,
    )

    for episode in range(n_episodes):

        tracker.new_episode()
        state = env.reset()
        # state = clip_state(state, n_clip)
        # state = cut_off_state(state, n_state)

        for _ in range(max_timesteps):

            action = mdqn.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            # next_state = clip_state(next_state, n_clip)
            # next_state = cut_off_state(next:state, n_state)

            tracker.track_reward(reward)
            mdqn.step(state, action, reward, next_state, done)
            state = next_state

            if done:
                print(
                    f"Episode: {episode} Reward: {tracker.episode_reward} Passengers {tracker.get_pick_ups()}"
                )
                break
        
        eps = round(max(eps * eps_decay, 0.001), 3)

    mdqn.save_model(log_path)
    tracker.plot(log_path)


def deploy_mdqn(n_episodes, wait):

    env_name = "Cabworld-v0"
    env = gym.make(env_name)

    n_states = 14
    n_actions = 6
    layer_size = 64
    buffer_size = 50000
    batch_size = 64
    eps_decay = 0.95
    eps = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mdqn = M_DQN_Agent(
        state_size=n_states,
        action_size=n_actions,
        layer_size=layer_size,
        BATCH_SIZE=batch_size,
        BUFFER_SIZE=buffer_size,
        LR=0.001,
        TAU=0.01,
        GAMMA=0.99,
        UPDATE_EVERY=10,
        device=device,
        seed=42,
    )

    current_folder = get_last_folder("mdqn")
    if not current_folder:
        print("No model")
        return
    current_model = os.path.join(current_folder, "mdqn.pth")
    print(current_model)
    mdqn.load_model(current_model)

    for _ in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = mdqn.act(state)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            env.render()
            time.sleep(wait)
            if done:
                print(f"Reward {episode_reward}")
                break
