from math import log
import os
import time
import logging
import numpy as np
from pyvirtualdisplay import Display
from collections import deque

import gym
import gym_cabworld

from algorithms.dqn_model import DQN, gen_epsilon_greedy_policy
from common.logging import create_log_folder, create_logger, get_last_folder
from common.logging import Tracker


# Fill buffer
episodes_without_training = 100


def train_dqn(n_episodes, munchhausen=False):

    disp = Display()
    disp.start()

    env_name = "Cabworld-v0"
    env = gym.make(env_name)

    n_states = env.observation_space.shape[1]
    n_actions = env.action_space.n
    n_hidden = 32

    lr = 0.001
    gamma = 0.975
    epsilon = 1
    epsilon_decay = 0.9975
    replay_size = 100
    target_update = 5

    log_path = create_log_folder("dqn")
    logger = create_logger(log_path)
    tracker = Tracker()

    dqn = DQN(n_states, n_actions, n_hidden, lr)

    memory = deque(maxlen=episodes_without_training * 1000)

    for episode in range(n_episodes + episodes_without_training):

        tracker.new_episode()

        if episode % target_update == 0:
            dqn.copy_target()

        policy = gen_epsilon_greedy_policy(dqn, epsilon, n_actions)
        state = env.reset()

        is_done = False
        steps = 0

        action = 0

        while not is_done:

            steps += 1
            action = policy(state)

            tracker.track_actions(state, action)

            next_state, reward, is_done, _ = env.step(action)
            tracker.track_reward(reward)
            memory.append((state, action, next_state, reward, is_done))

            if episode > episodes_without_training and steps % 10 == 0:
                if munchhausen:
                    dqn.replay_munchhausen(memory, replay_size, gamma)
                else:
                    dqn.replay(memory, replay_size, gamma)

            if is_done:
                print(
                    f"Episode: {episode} Reward: {tracker.episode_reward} Passengers {tracker.get_pick_ups()}"
                )
                if tracker.get_pick_ups() < 1:
                    for _ in range(1000):
                        memory.pop()

                break
            state = next_state

        if episode > episodes_without_training:
            epsilon = max(epsilon * epsilon_decay, 0.01)

        if episode > 0:
            tracker.track_epsilon(epsilon)

    dqn.save_model(log_path)
    tracker.plot(log_path)


def deploy_dqn(n_episodes, wait):

    env_name = "Cabworld-v0"
    env = gym.make(env_name)

    n_states = env.observation_space.shape[1]
    n_actions = env.action_space.n
    n_hidden = 32

    dqn = DQN(n_states, n_actions, n_hidden)

    current_folder = get_last_folder("dqn")
    if not current_folder:
        print("No model")
        return
    current_model = os.path.join(current_folder, "dqn.pth")
    print(current_model)
    dqn.load_model(current_model)

    for _ in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = dqn.deploy(state)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            env.render()
            time.sleep(wait)
            if done:
                print(f"Reward {episode_reward}")
                break
