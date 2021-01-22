import os
import time
import numpy as np

from collections import deque
from pyvirtualdisplay import Display

import gym
import gym_cabworld

from algorithms.dqn_model import DQN, gen_epsilon_greedy_policy
from common.features import clip_state, cut_off_state
from common.logging import create_log_folder, get_last_folder
from common.logging import Tracker


def train_ma_dqn(n_episodes):

    Display().start()
    env_name = "Cabworld-v1"
    env = gym.make(env_name)

    n_states = 14
    n_actions = 6
    n_hidden = 16

    lr = 0.01
    gamma = 0.99
    epsilon = 1
    epsilon_decay = 0.99
    replay_size = 100
    target_update = 5

    log_path = create_log_folder("ma-dqn")
    log_path2 = create_log_folder("ma-dqn")

    tracker = Tracker()
    tracker2 = Tracker()

    dqn = DQN(n_states, n_actions, n_hidden, lr)
    memory = deque(maxlen=50000)

    dqn2 = DQN(n_states, n_actions, n_hidden, lr)
    memory2 = deque(maxlen=50000)

    for episode in range(n_episodes):

        tracker.new_episode()
        tracker2.new_episode()

        if episode % target_update == 0:
            dqn.copy_target()
            dqn2.copy_target()

        policy = gen_epsilon_greedy_policy(dqn, epsilon, n_actions)
        policy2 = gen_epsilon_greedy_policy(dqn2, epsilon, n_actions)

        state, state2 = env.reset()
        # state = clip_state(state, n_clip)
        # state = cut_off_state(state, n_state)
        is_done = False
        steps = 0

        while not is_done:

            steps += 1

            action = policy(state)
            action2 = policy2(state2)

            (next_state, next_state2), (reward, reward2), is_done, _ = env.step(
                [action, action2]
            )
            # next_state = clip_state(next_state, n_clip)
            # next_state = cut_off_state(next_state, n_state)

            tracker.track_reward(reward)
            tracker2.track_reward(reward2)
            memory.append((state, action, next_state, reward, is_done))
            memory2.append((state2, action2, next_state2, reward2, is_done))

            if episode > 50 and steps % 100 == 0:
                dqn.replay(memory, replay_size, gamma)
                dqn2.replay(memory2, replay_size, gamma)

            if is_done:
                print(
                    f"Episode: {episode} Reward: {tracker.episode_reward} Passengers {tracker.get_pick_ups()}"
                )
                break
            state = next_state
            state2 = next_state2
        epsilon = max(epsilon * epsilon_decay, 0.01)

    dqn.save_model(log_path)
    dqn2.save_model(log_path2)
    tracker.plot(log_path)
    tracker2.plot(log_path2)


def deploy_dqn(n_episodes, wait):

    env_name = "Cabworld-v0"
    env = gym.make(env_name)

    n_states = 14
    n_actions = 6
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
