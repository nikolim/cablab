import os
import time
import numpy as np

from collections import deque
from pyvirtualdisplay import Display

import gym
import gym_cabworld

from algorithms.advnet import AdvNet

from algorithms.dqn_model import DQN, gen_epsilon_greedy_policy
from common.features import clip_state, cut_off_state, calc_shorter_way, add_msg_to_states, send_pos_to_other_cab, calc_adv_rewards
from common.logging import create_log_folder, get_last_folder
from common.logging import MultiTracker

# Fill buffer
episodes_without_training = 10

# Remove for deployment
# Display().start()


def train_ma_dqn(n_episodes, munchhausen=False):

    # Display().start()
    env_name = "Cabworld-v1"
    env = gym.make(env_name)

    n_agents = 2
    assert n_agents == 2 # ADVNET

    # calc additional msg signal
    n_states = 10
    n_actions = 7
    n_msg = 2
    n_hidden = 32

    lr = 0.001
    gamma = 0.975
    epsilon = 1
    epsilon_decay = 0.9975
    replay_size = 100
    target_update = 5

    log_path = create_log_folder("ma-dqn")
    tracker = MultiTracker(n_agents=n_agents)

    dqn_models = []
    adv_models = []
    memorys = []
    adv_memorys = []

    for _ in range(n_agents):
        dqn = DQN(n_states, n_actions, n_hidden, lr)
        adv = AdvNet()
        memory = deque(maxlen=50000)
        adv_memory = deque(maxlen=5000)

        dqn_models.append(dqn)
        memorys.append(memory)
        adv_memorys.append(adv_memory)
        adv_models.append(adv)

    for episode in range(n_episodes + episodes_without_training):

        tracker.new_episode()

        if episode % target_update == 0:
            for dqn in dqn_models:
                dqn.copy_target()

        policies = []
        adv_policies = []
        for i in range(n_agents):

            policy = gen_epsilon_greedy_policy(
                dqn_models[i], epsilon, n_actions)
            policies.append(policy)

            adv_policy = gen_epsilon_greedy_policy(
                adv_models[i], epsilon, n_msg)
            adv_policies.append(adv_policy)

        states = env.reset()

        # ADV
        adv_inputs = send_pos_to_other_cab(states)
        msgs = []   
        for i in range(n_agents): 
            msgs.append(adv_policies[i](adv_inputs[i]))
        states = add_msg_to_states(states, msgs)

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

            adv_rewards = calc_adv_rewards(adv_inputs, msgs)
            
            tracker.track_reward(rewards)
            tracker.track_actions(actions)

            # ADV
            adv_inputs_next = send_pos_to_other_cab(next_states)
            msgs_next = []   
            for i in range(n_agents): 
                msgs_next.append(adv_policies[i](adv_inputs[i]))
            next_states = add_msg_to_states(next_states, msgs_next)

            for i in range(n_agents):

                memorys[i].append(
                    (states[i], actions[i], next_states[i], rewards[i], is_done))

                adv_memorys[i].append((adv_inputs[i], msgs[i], adv_rewards[i]))

                for i in range(n_agents):

                    if episode > episodes_without_training and steps % 10 == 0:
                        if munchhausen:
                            dqn_models[i].replay_munchhausen(
                                memorys[i], replay_size, gamma)
                        else:
                            dqn_models[i].replay(
                                memorys[i], replay_size, gamma)
                        adv_models[i].replay(adv_memorys[i], replay_size)


            if is_done:
                print(
                    f"Episode: {episode} Reward: {tracker.episode_reward} Passengers {tracker.get_pick_ups()} Do-nothing {tracker.do_nothing}"
                )
                break
            
            states = next_states
            adv_inputs = adv_inputs_next
            msgs = msgs_next

        if episode > episodes_without_training:
            epsilon = max(epsilon * epsilon_decay, 0.01)

    for i in range(n_agents):
        dqn_models[i].save_model(log_path, number=str(i+1))
    tracker.plot(log_path)


def deploy_ma_dqn(n_episodes, wait):

    env_name = "Cabworld-v1"
    env = gym.make(env_name)

    n_agents = 2
    n_states = 10
    n_actions = 7
    n_hidden = 32

    # current_folder = get_last_folder("ma-dqn")
    current_folder = get_last_folder("ma-dqn")
    if not current_folder:
        print("No model")
        return

    dqn_models = []
    for i in range(n_agents):
        dqn = DQN(n_states, n_actions, n_hidden)
        dqn_models.append(dqn)
        current_model = os.path.join(current_folder, "dqn" + str(i+1) + ".pth")
        # current_model = os.path.join(current_folder, "dqn.pth")
        dqn_models[i].load_model(current_model)

    for _ in range(n_episodes):
        states = env.reset()
        states = add_msg_to_states(states)
        episode_reward = 0
        done = False
        while not done:
            actions = [dqn.deploy(state)
                       for dqn, state in zip(dqn_models, states)]
            states, rewards, done, _ = env.step(actions)
            states = add_msg_to_states(states)
            episode_reward += sum(rewards)
            env.render()
            time.sleep(wait)
            if done:
                print(f"Reward {episode_reward}")
                break
