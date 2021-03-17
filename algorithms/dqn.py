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
from common.features import extend_single_agent_state

# Fill buffer
episodes_without_training = 500


def train_dqn(n_episodes, munchhausen=False, extended=True):

    disp = Display()
    disp.start()

    env_name = "Cabworld-v0"
    env = gym.make(env_name)

    # n_states = env.observation_space.shape[1] + (1 if extended else 0)
    n_states = 11
    n_actions = env.action_space.n 
    n_hidden = 64

    lr = 0.001
    gamma = 0.975
    epsilon = 1
    epsilon_decay = 0.9975
    replay_size = 100
    target_update = 50

    log_path = create_log_folder("dqn")
    logger = create_logger(log_path)
    tracker = Tracker(logger)

    tracker.write_to_log("Epsiodes " + str(n_episodes))
    tracker.write_to_log("Episodes without training " + str(episodes_without_training))
    tracker.write_to_log("Hidden neurons " + str(n_hidden))
    tracker.write_to_log("Learning Rate " + str(lr))
    tracker.write_to_log("Gamma " + str(gamma))
    tracker.write_to_log("Epsilon initial  " + str(epsilon))
    tracker.write_to_log("Epsilon decay " + str(epsilon_decay))
    tracker.write_to_log("Replay size " + str(replay_size))
    tracker.write_to_log("Target update " + str(target_update))
    
    dqn = DQN(n_states, n_actions, n_hidden, lr)

    memory = deque(maxlen=episodes_without_training * 1000)

    for episode in range(n_episodes + episodes_without_training):

        tracker.new_episode()

        if episode % target_update == 0:
            dqn.copy_target()

        policy = gen_epsilon_greedy_policy(dqn, epsilon, n_actions)
        state = env.reset()

        tracker.save_dest_to_passengers(state)

        # state = extend_single_agent_state(state) if extended else state

        is_done = False
        steps = 0

        action = 0

        while not is_done:

            steps += 1
            action = policy(state)

            tracker.track_actions(state, action)

            next_state, reward, is_done, _ = env.step(action)
            # next_state = (
            #     extend_single_agent_state(next_state) if extended else next_state
            # )
            tracker.track_reward(reward, action, state, next_state)

            memory.append((state, action, next_state, reward, is_done))

            if episode > episodes_without_training and steps % 10 == 0:
                if munchhausen:
                    dqn.replay_munchhausen(memory, replay_size, gamma)
                else:
                    dqn.replay(memory, replay_size, gamma)

            if is_done:
                if tracker.opt_passenger > 0:
                    ratio = round(tracker.opt_passenger/  tracker.pick_ups, 3)
                elif tracker.pick_ups > 0: 
                    ratio = 0
                else: 
                    ratio = 0.5
                print(
                    f"Episode: {episode} Reward: {tracker.episode_reward} Passengers {tracker.get_pick_ups()} Opt: {tracker.get_opt_pick_ups()} Ratio: {ratio}"
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


def deploy_dqn(n_episodes, wait, extended=True):

    env_name = "Cabworld-v0"
    env = gym.make(env_name)

    n_states = env.observation_space.shape[1] + (1 if extended else 0)
    n_actions = env.action_space.n
    n_hidden = 64

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
        state = extend_single_agent_state(state) if extended else state
        episode_reward = 0
        done = False
        while not done:
            action = dqn.deploy(state)
            state, reward, done, _ = env.step(action)
            state = extend_single_agent_state(state) if extended else state
            episode_reward += reward
            env.render()
            time.sleep(wait)
            if done:
                print(f"Reward {episode_reward}")
                break
