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
    add_old_assignment,
    assign_passenger,
    clip_state,
    create_adv_inputs,
    cut_off_state,
    calc_shorter_way,
    add_fixed_msg_to_states,
    add_msg_to_states,
    picked_up_assigned_psng,
    random_assignment,
    optimal_assignment,
    send_pos_to_other_cab,
    calc_adv_rewards,
)
from common.logging import create_log_folder, create_logger, get_last_folder
from common.logging import MultiTracker

# Fill buffer
episodes_without_training = 100

# ADV = calculated communication with fixed protocoll
# COMM = predefined communication

assign_passenger = "random"
assign_passenger = "opt"

def train_ma_dqn(n_episodes, munchhausen=False, adv=False, comm=False):

    disp = Display()
    disp.start()

    env_name = "Cabworld-v3"
    env = gym.make(env_name)

    n_agents = env.observation_space.shape[0]

    if adv:
        episodes_only_adv = 1000
    else:
        episodes_only_adv = 0

    # calc additional msg signal
    n_states = env.observation_space.shape[1] + (1 if (adv or comm) else 0) #+ 1
    n_actions = env.action_space.n
    n_msg = 2
    n_hidden = 64

    lr = 0.001
    gamma = 0.9
    epsilon = 1
    adv_epsilon = 1
    #epsilon_decay = 0.9985 # working
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

    
    dqn = DQN(n_states, n_actions, n_hidden, lr)
    memory = deque(maxlen=episodes_without_training * 1000)

    if adv:
        adv = AdvNet(n_input=n_agents*4, n_msg=n_agents)
        adv_memory = deque(maxlen=1000)

    for episode in range(n_episodes + episodes_without_training + episodes_only_adv):

        tracker.new_episode()
        tracker.reset_waiting_time(log=False)

        # temporary pick up and drop off counter
        n_pick_ups = 0
        n_drop_offs = 0

        if episode % target_update == 0:
            dqn.copy_target()

        policy = gen_epsilon_greedy_policy(
            dqn, epsilon, n_actions)

        if adv:
            adv_policy = gen_epsilon_greedy_policy(
                adv, adv_epsilon, n_msg
            )

        states = env.reset()

        if adv:
            adv_inputs = create_adv_inputs(states)
            assignment = adv_policy(adv_inputs)
            msgs = [0,1] if assignment == 0 else [1,0]
            states = add_msg_to_states(states, msgs)
            
        is_done = False
        steps = 0

        while not is_done:

            steps += 1

            actions = []

            for i in range(n_agents):
                actions.append(policy(states[i]))

            next_states, rewards, is_done, _ = env.step(actions)
            
            for reward, action in zip(rewards, actions): 
                if reward == 1:
                    if action == 4: 
                        n_pick_ups += 1 
                    else: 
                        n_drop_offs += 1

            if n_pick_ups == 2: 
                waiting_time = tracker.reset_waiting_time()
                adv_reward = -(waiting_time /1000)
                tracker.track_adv_reward(adv_reward)
                n_pick_ups = 0
                if adv:
                    adv_memory.append((adv_inputs, assignment, adv_reward))

            if n_drop_offs == 2: 
                tracker.reset_waiting_time(log=False)
                n_drop_offs = 0
                if adv:
                    adv_inputs = create_adv_inputs(next_states)
                    assignment = adv_policy(adv_inputs)
                    msgs = [0,1] if assignment == 0 else [1,0]
                    next_states = add_msg_to_states(next_states, msgs)
            else: 
                if adv:
                    next_states = add_old_assignment(next_states, states)

            tracker.track_reward_and_action(rewards, actions, states)

            # give only reward for pick-up if passenger was assigned
            for reward, action, state, i in zip(rewards, actions, states, list(range(n_agents))): 
                if action == 4 and reward == 1: 
                    if picked_up_assigned_psng(state): 
                        rewards[i] = 2
                    else: 
                        rewards[i] = 0 

            #summed_rewards = sum(rewards)

            for i in range(n_agents):
                assert len(next_states[0]) == 12
                memory.append(
                    (states[i], actions[i],
                     next_states[i], rewards[i], is_done) #rewards[i]
                )

                if episode > episodes_without_training and steps % 10 == 0:

                    if episode > (episodes_without_training + n_episodes):
                        adv.replay(adv_memory, 10)

                    if episode < (episodes_without_training + n_episodes):
                        if munchhausen:
                            dqn.replay_munchhausen(
                                memory, replay_size, gamma
                            )
                        else:
                            dqn.replay(
                                memory, replay_size, gamma)

            if is_done:
                adv_rewards = f"ADV {tracker.adv_episode_rewards}" if adv else ""
                print(
                    f"Episode: {episode} Reward: {tracker.get_rewards()} Passengers {tracker.get_pick_ups()} Do-nothing {tracker.get_do_nothing()} {adv_rewards}"
                )

                # selective pops
                if sum(tracker.get_pick_ups()) < 1:
                    for _ in range(n_agents * 1000):
                        memory.pop()
                break

            states = next_states

        if episode > episodes_without_training:
            adv_epsilon = max(adv_epsilon * adv_epsilon_decay, 0.01)

        if episode > (episodes_without_training + n_episodes):
            epsilon = max(epsilon * epsilon_decay, 0.01)


    for i in range(n_agents):
        dqn.save_model(log_path, number=str(i + 1))
        if adv:
            adv.save_model(log_path, number=str(i + 1))

    tracker.plot(log_path)


def deploy_ma_dqn(n_episodes, wait, adv=False, comm=False, render=False, eval=False):

    env_name = "Cabworld-v3"
    env = gym.make(env_name)

    if not render: 
        disp = Display()
        disp.start()

    n_agents = env.observation_space.shape[0]
    n_states = env.observation_space.shape[1] + (1 if (adv or comm) else 0) + 1
    n_actions = env.action_space.n
    n_hidden = 64

    tracker = MultiTracker(n_agents=n_agents)

    #current_folder = get_last_folder("dqn")
    current_folder = get_last_folder("ma-dqn")
    if not current_folder:
        print("No model")
        return

    current_folder = "/home/niko/Info/cablab/runs/ma-dqn/202"

    dqn_models = []
    adv_models = []

    for i in range(n_agents):
        dqn = DQN(n_states, n_actions, n_hidden)
        dqn_models.append(dqn)
        current_model = os.path.join(
            current_folder, "dqn1.pth")
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
        states = random_assignment(states)
        #states = optimal_assignment(states)

        n_pick_ups = 0

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
            old_states = states
            states, rewards, done, _ = env.step(actions)

            for reward, action in zip(rewards, actions): 
                if reward == 1 and action == 4: 
                    n_pick_ups += 1 

            if n_pick_ups == 2: 
                tracker.reset_waiting_time()
                n_pick_ups = 0
                states = random_assignment(states)
                #states = optimal_assignment(states)
            else: 
                states = add_old_assignment(states, old_states)
                pass

            #tracker.track_reward(rewards)
            #tracker.track_actions(states, actions, comm=comm)
            tracker.track_reward_and_action(rewards, actions, states)

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
                time.sleep(wait)
            if done:
                print(
                    f"Episode: {episode} Reward: {tracker.get_rewards()} Passengers {tracker.get_pick_ups()} Do-nothing {tracker.get_do_nothing()}"
                )
                break

    if eval:
        tracker.plot(log_path=current_folder, eval=True)

    