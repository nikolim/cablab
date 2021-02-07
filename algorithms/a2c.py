import os
import time
import numpy as np
import torch

import gym
import gym_cabworld

from algorithms.a2c_model import PolicyNetwork
from common.features import clip_state, cut_off_state
from common.logging import create_log_folder, get_last_folder
from common.logging import Tracker


def train_a2c(n_episodes):

    from pyvirtualdisplay import Display
    Display().start()

    env_name = "Cabworld-v0"
    env = gym.make(env_name)

    n_states = 14
    n_actions = 6
    max_timesteps = 1000
    gamma = 0.99
    rewards = []

    log_path = create_log_folder("a2c")
    tracker = Tracker()

    a2c = PolicyNetwork(n_states, n_actions)

    for episode in range(n_episodes):

        tracker.new_episode()
        state = env.reset()
        # state = clip_state(state, n_clip)
        # state = cut_off_state(state, n_state)

        log_probs = []
        state_values = []

        for _ in range(max_timesteps):

            action, log_prob, state_value = a2c.get_action(state)
            state, reward, done, _ = env.step(action)
            # state = clip_state(state, n_clip)
            # state = cut_off_state(state, n_state)

            tracker.track_reward(reward)
            log_probs.append(log_prob)
            state_values.append(state_value)
            rewards.append(reward)

            if done:
                print(
                    f"Episode: {episode} Reward: {tracker.episode_reward} Passengers {tracker.get_pick_ups()}"
                )
                returns = []
                Gt = 0
                pw = 0
                for reward in rewards[::-1]:
                    Gt += gamma ** pw * reward
                    pw += 1
                    returns.append(Gt)

                returns = returns[::-1]
                returns = torch.tensor(returns)
                returns = (returns - returns.mean()) / (returns.std() + 1e-9)
                a2c.update(returns, log_probs, state_values, episode)
                break

    a2c.save_model(log_path)
    tracker.plot(log_path)


def deploy_a2c(n_episodes, wait):

    env_name = "Cabworld-v0"
    env = gym.make(env_name)

    n_states = 14
    n_actions = 6

    a2c = PolicyNetwork(n_states, n_actions)

    current_folder = get_last_folder("a2c")
    if not current_folder:
        print("No model")
        return
    current_model = os.path.join(current_folder, "a2c.pth")
    print(current_model)
    a2c.load_model(current_model)

    for _ in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = a2c.get_action(state)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            env.render()
            time.sleep(wait)
            if done:
                print(f"Reward {episode_reward}")
                break
