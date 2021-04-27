import time
import random
import numpy as np

import gym
import gym_cabworld

from common.logging import create_log_folder
from common.logging import Tracker


def random_policy(state):
    state = list(state)
    if state[4] == 1:
        return 4
    if state[5] == 1:
        return 5
    else:
        move_flags = state[:4]
        legal_moves = [i for i, flag in enumerate(move_flags) if flag == 1]
        return random.sample(legal_moves, 1)[0]


def train_random(n_episodes):

    from pyvirtualdisplay import Display

    Display().start()

    env_name = "Cabworld-v0"
    env = gym.make(env_name)

    max_timesteps = 1000

    log_path = create_log_folder("rand")
    tracker = Tracker()

    for episode in range(n_episodes):

        tracker.new_episode()
        state = env.reset()

        for _ in range(max_timesteps):

            action = random_policy(state)
            state, reward, done, _ = env.step(action)
            tracker.track_reward(reward)

            if done:
                print(
                    f"Episode: {episode} Reward: {tracker.episode_reward} Passengers {tracker.get_pick_ups()}"
                )
                break

    tracker.plot(log_path)


def deploy_random(n_episodes, wait):

    env_name = "Cabworld-v0"
    env = gym.make(env_name)

    for _ in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = random_policy(state)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            env.render()
            time.sleep(wait)
            if done:
                print(f"Reward {episode_reward}")
                break
