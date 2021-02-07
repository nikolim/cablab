import os
import time
import numpy as np

from collections import deque

import gym
import gym_cabworld

from algorithms.dqn_model import DQN, gen_epsilon_greedy_policy
from common.features import clip_state, cut_off_state
from common.logging import create_log_folder, get_last_folder
from common.logging import Tracker

from common.features import calc_potential

n_states = 13
n_actions = 6
n_hidden = 32

# Fill buffer
episodes_without_training = 50

def train_dqn(n_episodes, munchhausen=False):

    from pyvirtualdisplay import Display
    Display().start()

    env_name = "Cabworld-v0"
    env = gym.make(env_name)

    lr = 0.01
    gamma = 0.99
    epsilon = 1
    epsilon_decay = 0.99
    replay_size = 100
    target_update = 5

    log_path = create_log_folder("dqn")
    tracker = Tracker()

    dqn = DQN(n_states, n_actions, n_hidden, lr)
    memory = deque(maxlen=50000)

    for episode in range(n_episodes + episodes_without_training):

        tracker.new_episode()

        if episode % target_update == 0:
            dqn.copy_target()

        policy = gen_epsilon_greedy_policy(dqn, epsilon, n_actions)
        state = env.reset()
        # state = clip_state(state, n_clip)
        # state = cut_off_state(state, n_state)

        tracker.save_dest_to_passengers(state)

        is_done = False
        steps = 0

        action = 0

        while not is_done:

            steps += 1
            action = policy(state)

            next_state, reward, is_done, _ = env.step(action)
            # next_state = clip_state(next_state, n_clip)
            # next_state = cut_off_state(next_state, n_state)

            tracker.track_reward(reward, action, state, next_state)

            reward += calc_potential(state, state, gamma)

            memory.append((state, action, next_state, reward, is_done))

            if munchhausen:
                if episode > episodes_without_training and steps % 10 == 0:
                    dqn.replay_munchhausen(memory, replay_size * 10, gamma)
            else: 
                if episode > episodes_without_training and steps % 100 == 0:
                    dqn.replay(memory, replay_size, gamma)

            if is_done:
                print(
                    f"Episode: {episode} Reward: {tracker.episode_reward} Passengers {tracker.get_pick_ups()} Opt: {tracker.get_opt_pick_ups()}"
                )
                break
            state = next_state
        if episode > episodes_without_training:
            epsilon = max(epsilon * epsilon_decay, 0.01)

    dqn.save_model(log_path)
    tracker.plot(log_path)


def deploy_dqn(n_episodes, wait):

    env_name = "Cabworld-v0"
    env = gym.make(env_name)
    dqn = DQN(n_states, n_actions)

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
