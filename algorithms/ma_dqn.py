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
from common.logging import MultiTracker

# Fill buffer
episodes_without_training = 50

# Remove for deployment
Display().start()


def train_ma_dqn(n_episodes, munchhausen=False):

    # Display().start()
    env_name = "Cabworld-v1"
    env = gym.make(env_name)

    n_agents = 3
    n_states = 9
    n_actions = 7
    n_hidden = 16

    lr = 0.01
    gamma = 0.99
    epsilon = 1
    epsilon_decay = 0.99
    replay_size = 100
    target_update = 5

    log_path = create_log_folder("ma-dqn")
    tracker = MultiTracker(n_agents=n_agents)

    dqn_models = []
    memorys = []
    
    for _ in range(n_agents): 
        dqn = DQN(n_states, n_actions, n_hidden, lr)
        memory = deque(maxlen=50000)
        dqn_models.append(dqn)
        memorys.append(memory)


    for episode in range(n_episodes):

        tracker.new_episode()

        if episode % target_update == 0:
            for dqn in dqn_models:
                dqn.copy_target()

        policies = []
        for i in range(n_agents): 
            policy = gen_epsilon_greedy_policy(dqn_models[i], epsilon, n_actions)
            policies.append(policy)
            
        states = env.reset()
        # state = clip_state(state, n_clip)
        # state = cut_off_state(state, n_state)
        is_done = False
        steps = 0

        while not is_done:

            steps += 1
            
            actions = []
            for i in range(n_agents): 
                actions.append(policies[i](states[i]))

            next_states, rewards, is_done, _ = env.step(
                actions
            )
            # next_state = clip_state(next_state, n_clip)
            # next_state = cut_off_state(next_state, n_state)

            tracker.track_reward(rewards)
            tracker.track_actions(actions)

            for i in range(n_agents):

                reward = 1000 if rewards[i] == 100 else rewards[i]
                memorys[i].append((states[i], actions[i], next_states[i], reward, is_done))

                for i in range(n_agents):
                    if munchhausen:
                        if episode > episodes_without_training and steps % 10 == 0:
                            dqn_models[i].replay_munchhausen(memorys[i], replay_size, gamma)
                    else:
                        if episode > episodes_without_training and steps % 10 == 0:
                            dqn_models[i].replay(memorys[i], replay_size, gamma)
                     

            if is_done:
                print(
                    f"Episode: {episode} Reward: {tracker.episode_reward} Passengers {tracker.get_pick_ups()} Do-nothing {tracker.do_nothing}"
                )
                break
            states = next_states
        if episode > episodes_without_training:
            epsilon = max(epsilon * epsilon_decay, 0.01)

    for i in range(n_agents):
        dqn_models[i].save_model(log_path, number=str(i+1))
    tracker.plot(log_path)


def deploy_ma_dqn(n_episodes, wait):

    env_name = "Cabworld-v1"
    env = gym.make(env_name)

    n_agents = 3
    n_states = 14
    n_actions = 7

    current_folder = get_last_folder("ma-dqn")
    # current_folder = get_last_folder("dqn")
    if not current_folder:
        print("No model")
        return

    dqn_models = []
    for i in range(n_agents):
        dqn = DQN(n_states, n_actions)
        dqn_models.append(dqn)
        current_model = os.path.join(current_folder, "dqn" + str(i+1) + ".pth")
        # current_model = os.path.join(current_folder, "dqn.pth")
        dqn_models[i].load_model(current_model)

    for _ in range(n_episodes):
        states = env.reset()
        episode_reward = 0
        done = False
        while not done:
            actions = [dqn.deploy(state) for dqn, state in zip(dqn_models, states)]
            states, rewards, done, _ = env.step(actions)
            episode_reward += sum(rewards)
            env.render()
            time.sleep(wait)
            if done:
                print(f"Reward {episode_reward}")
                break
