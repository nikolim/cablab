import os
import time
import numpy as np

from collections import deque
from pyvirtualdisplay import Display

import gym
import gym_cabworld

from algorithms.advnet import AdvNet

from algorithms.dqn_model import DQN, gen_epsilon_greedy_policy
from common.features import (
    clip_state,
    cut_off_state,
    calc_shorter_way,
    add_fixed_msg_to_states,
    add_msg_to_states,
    send_pos_to_other_cab,
    calc_adv_rewards,
)
from common.logging import create_log_folder, create_logger, get_last_folder
from common.logging import MultiTracker

# Fill buffer
episodes_without_training = 500

# ADV = calculated communication with fixed protocoll
# COMM = predefined communication

def train_ma_dqn(n_episodes, munchhausen=False, adv=False, comm=False):

    disp = Display()
    disp.start()

    env_name = "Cabworld-v1"
    env = gym.make(env_name)

    n_agents = env.observation_space.shape[0]

    if adv:
        assert n_agents == 2  # ADVNET
        episodes_only_adv = 500
    else:
        episodes_only_adv = 0

    # calc additional msg signal
    n_states = env.observation_space.shape[1] + (1 if (adv or comm) else 0)
    n_actions = env.action_space.n
    n_msg = 2
    n_hidden = 64

    lr = 0.001
    gamma = 0.975
    epsilon = 1
    adv_epsilon = 1
    epsilon_decay = 0.9975
    adv_epsilon_decay = 0.99

    replay_size = 100
    target_update = 50

    log_path = create_log_folder("ma-dqn")
    logger = create_logger(log_path)
    tracker = MultiTracker(n_agents=n_agents, logger=logger)

    tracker.write_to_log("Epsiodes " + str(n_episodes))
    tracker.write_to_log("Episodes without training " + str(episodes_without_training))
    tracker.write_to_log("Munchhausen " + str(munchhausen))
    tracker.write_to_log("Predefined communication " + str(comm))
    tracker.write_to_log("Adv communication " + str(adv))
    tracker.write_to_log("Hidden neurons " + str(n_hidden))
    tracker.write_to_log("Learning Rate " + str(lr))
    tracker.write_to_log("Gamma " + str(gamma))
    tracker.write_to_log("Epsilon initial  " + str(epsilon))
    tracker.write_to_log("Epsilon decay " + str(epsilon_decay))
    tracker.write_to_log("Replay size " + str(replay_size))
    tracker.write_to_log("Target update " + str(target_update))

    dqn_models = []
    memorys = []

    if adv:
        adv_models = []
        adv_memorys = []

    for _ in range(n_agents):
        dqn = DQN(n_states, n_actions, n_hidden, lr)
        memory = deque(maxlen=episodes_without_training * 1000)
        dqn_models.append(dqn)
        memorys.append(memory)

        if adv:
            adv = AdvNet()
            adv_memory = deque(maxlen=episodes_without_training * 1000)
            adv_memorys.append(adv_memory)
            adv_models.append(adv)

    for episode in range(n_episodes + episodes_without_training + episodes_only_adv):

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

            if adv:
                adv_policy = gen_epsilon_greedy_policy(
                    adv_models[i], adv_epsilon, n_msg
                )
                adv_policies.append(adv_policy)

        states = env.reset()

        if adv:
            adv_inputs = send_pos_to_other_cab(states)
            msgs = []
            for i in range(n_agents):
                msgs.append(adv_policies[i](adv_inputs[i]))
            states = add_msg_to_states(states, msgs)
        elif comm:
            states = add_fixed_msg_to_states(states)

        is_done = False
        steps = 0

        while not is_done:

            steps += 1

            actions = []
            for i in range(n_agents):
                # also remove tag
                state = list(states[i])
                #state[-1] = 0
                state = tuple(state)
                actions.append(policies[i](state))

            next_states, rewards, is_done, _ = env.step(actions)

            tracker.track_reward(rewards)
            tracker.track_actions(states, actions, comm=(comm or adv))

            if adv:
                adv_rewards = calc_adv_rewards(adv_inputs, msgs)
                tracker.track_adv_reward(adv_rewards)

            if adv:
                adv_inputs_next = send_pos_to_other_cab(next_states)
                msgs_next = []
                for i in range(n_agents):
                    msgs_next.append(adv_policies[i](adv_inputs[i]))
                next_states = add_msg_to_states(next_states, msgs_next)
            elif comm:
                next_states = add_fixed_msg_to_states(next_states)

            for i in range(n_agents):

                # artificial set last entry to zero after tracking
                state = list(states[i])
                #state[-1] = 0
                state = tuple(state)

                next_state = list(next_states[i])
                #next_state[-1] = 0
                next_state = tuple(next_state)

                memorys[i].append(
                    (state, actions[i],
                     next_state, rewards[i], is_done)
                )
                if adv: 
                    adv_memorys[i].append(
                        (adv_inputs[i], msgs[i], adv_rewards[i]))

                if episode > episodes_without_training and steps % 10 == 0:

                    if adv:
                        adv_models[i].replay(adv_memorys[i], replay_size)

                    if episode > (episodes_without_training + episodes_only_adv):
                        if munchhausen:
                            dqn_models[i].replay_munchhausen(
                                memorys[i], replay_size, gamma
                            )
                        else:
                            dqn_models[i].replay(
                                memorys[i], replay_size, gamma)

            if is_done:
                adv_rewards = f"ADV {tracker.adv_episode_rewards}" if adv else ""
                print(
                    f"Episode: {episode} Reward: {tracker.get_rewards()} Passengers {tracker.get_pick_ups()} Do-nothing {tracker.get_do_nothing()} {adv_rewards}"
                )

                # selective pops
                for i, pick_ups in enumerate(tracker.get_pick_ups()): 
                    if pick_ups < 1: 
                        for _ in range(1000): 
                            memorys[i].pop()
                
                break

            states = next_states

            if adv:
                adv_inputs = adv_inputs_next
                msgs = msgs_next

        if episode > episodes_without_training:
            adv_epsilon = max(adv_epsilon * adv_epsilon_decay, 0.01)

        if episode > (episodes_without_training + episodes_only_adv):
            epsilon = max(epsilon * epsilon_decay, 0.01)

        if episode > 0:
            tracker.track_epsilon(epsilon)

    for i in range(n_agents):
        dqn_models[i].save_model(log_path, number=str(i + 1))
        if adv:
            adv_models[i].save_model(log_path, number=str(i + 1))

    tracker.plot(log_path)


def deploy_ma_dqn(n_episodes, wait, adv=False, comm=False, render=False):

    env_name = "Cabworld-v1"
    env = gym.make(env_name)

    if not render: 
        disp = Display()
        disp.start()

    n_agents = env.observation_space.shape[0]
    n_states = env.observation_space.shape[1] + (1 if (adv or comm) else 0)
    n_actions = env.action_space.n
    n_hidden = 64

    tracker = MultiTracker(n_agents=n_agents)

    #current_folder = get_last_folder("dqn")
    current_folder = get_last_folder("ma-dqn")
    if not current_folder:
        print("No model")
        return

    dqn_models = []
    adv_models = []

    for i in range(n_agents):
        dqn = DQN(n_states, n_actions, n_hidden)
        dqn_models.append(dqn)
        current_model = os.path.join(
            current_folder, "dqn" + str(i + 1) + ".pth")
        #current_model = "/home/niko/Info/cablab/runs/dqn/148/dqn.pth"
        dqn_models[i].load_model(current_model)

        if adv:
            adv = AdvNet()
            adv_models.append(adv)
            current_adv_model = os.path.join(
                current_folder, "adv" + str(i + 1) + ".pth"
            )
            adv_models[i].load_model(current_adv_model)

    for episode in range(n_episodes):

        tracker.new_episode()
        states = env.reset()

        if adv:
            adv_inputs = send_pos_to_other_cab(states)
            msgs = []
            for i in range(n_agents):
                msgs.append(adv_models[i].deploy((adv_inputs[i])))
            states = add_msg_to_states(states, msgs)
        elif comm:
            states = add_fixed_msg_to_states(states)

        done = False
        while not done:
            actions = [dqn.deploy(state)
                       for dqn, state in zip(dqn_models, states)]
            states, rewards, done, _ = env.step(actions)

            tracker.track_reward(rewards)
            tracker.track_actions(states, actions, comm=comm)
            
            if adv:
                adv_inputs = send_pos_to_other_cab(states)
                msgs = []
                for i in range(n_agents):
                    msgs.append(adv_models[i].deploy((adv_inputs[i])))
                states = add_msg_to_states(states, msgs)
            elif comm:
                states = add_fixed_msg_to_states(states)

            if render:
                env.render()
            if render:
                time.sleep(wait)
            if done:
                print(
                    f"Episode: {episode} Reward: {tracker.get_rewards()} Passengers {tracker.get_pick_ups()} Do-nothing {tracker.get_do_nothing()}"
                )
                break

        if episode > 0:
                tracker.track_epsilon(0.01)

    