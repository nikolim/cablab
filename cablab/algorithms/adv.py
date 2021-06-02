import os
import time
import toml
import shutil
import random
import numpy as np
from pathlib import Path
from collections import deque
from pyvirtualdisplay import Display

import gym
import gym_cabworld

from cablab.algorithms.adv_model import *
from cablab.algorithms.dqn_model import *
from cablab.common.features import *
from cablab.common.logging import *


abs_path = Path(__file__).parent.absolute()
madqn_cfg_file = "ma_dqn_conf.toml"
madqn_cfg_file_path = os.path.join(abs_path, "../configs", madqn_cfg_file)
madqn_cfg_file_path = "/home/niko/Info/cablab/runs/ma-dqn/6/ma_dqn_conf.toml"

def train_adv(n_episodes, version):
    """
    Training only Adv on a pretrained models
    """
    
    # Start virtual display
    disp = Display()
    disp.start()

    cfg = toml.load(open(madqn_cfg_file_path), _dict=dict)
    print(f'Config loaded: {madqn_cfg_file_path}')

    env_name = 'Cabworld-' + version
    env = gym.make(env_name)

    n_agents = env.observation_space.shape[0]
    n_states = env.observation_space.shape[1] + 1
    n_actions = env.action_space.n

    # If single or multiple adv-policies should be trained
    n_policies = n_agents if cfg['decentralized'] else 1

    log_path = create_log_folder("adv")
    tracker = MultiTracker(n_agents=n_agents, version=version)

    # copy config-file into log-folder
    shutil.copyfile(madqn_cfg_file_path,
                    os.path.join(log_path, madqn_cfg_file))

    dqn = DQN(n_states, n_actions, cfg)
    # Select model trained on assigned passengers (single or multi-agent)
    current_model = "/home/niko/Info/cablab/runs/ma-dqn/6/dqn1.pth"
    dqn.load_model(current_model)

    advs = []
    n_input = 6 if version == 'v2' else 8
    for _ in range(n_policies):
        advs.append(AdvNet(n_input=n_input, n_msg=n_agents, lr=cfg['adv_lr']))

    adv_memory = deque(maxlen=cfg['adv_memory_size'])
    consens_arr = []

    for episode in range(n_episodes + cfg['adv_eps_without_training']):

        log_episode = episode >= cfg['adv_eps_without_training']
        tracker.new_episode(log_episode)
        tracker.reset_waiting_time()

        # temp episode counter
        n_pick_ups = 0
        n_drop_offs = 0
        equal_assignments = 0
        different_assignments = 0

        adv_policies = []
        for adv in advs:
            adv_policies.append(gen_epsilon_greedy_policy(
                adv, cfg['adv_epsilon'], cfg['n_msg']))

        states = env.reset()

        if version == 'v2':
            adv_inputs = create_adv_inputs_single(states)
        else:
            adv_inputs = create_adv_inputs(states)

        assignments = []
        for adv_policy in adv_policies:
            assignments.append(adv_policy(adv_inputs))

        # check if agents agree on assignment
        if cfg['decentralized']:
            if assignments.count(assignments[0]) == len(assignments):
                equal_assignments += 1
            else:
                different_assignments += 1

        assignment = random.sample(assignments, 1)[0]
        msgs = [0, 1] if assignment == 0 else [1, 0]
        states = add_msg_to_states(states, msgs)

        is_done = False
        steps = 0

        while not is_done:

            steps += 1
            actions = []

            for i in range(n_agents):
                actions.append(dqn.deploy(states[i]))

            next_states, rewards, is_done, _ = env.step(actions)
            tracker.track_reward_and_action(rewards, actions, states)

            if version == 'v2':
                if passenger_spawn(states[0], next_states[0]):
                    tracker.reset_action_counter()

            for reward, action in zip(rewards, actions):
                if reward == 1:
                    if action == 4:
                        n_pick_ups += 1
                        tracker.add_waiting_time()
                    else:
                        n_drop_offs += 1

            # v2 specific logic
            if version == 'v2':
                if n_pick_ups == 1:
                    n_pick_ups = 0
                    # take actions taken from spawn until pickup as feedback for ADV
                    adv_reward = -tracker.reset_action_counter()
                    tracker.track_adv_reward(adv_reward)
                    adv_memory.append((adv_inputs, assignment, adv_reward))

                if passenger_spawn(states[0], next_states[0]):
                    # assign passenger randomly after new passenger is spawn
                    adv_inputs = create_adv_inputs_single(next_states)
                    assignment = adv_policy(adv_inputs)
                    msgs = [0, 1] if assignment == 0 else [1, 0]
                    next_states = add_msg_to_states(next_states, msgs)
                else:
                    next_states = add_old_assignment(next_states, states)

            # v3 specific logic
            if version == 'v3':
                if n_pick_ups == 2:
                    waiting_time = tracker.reset_waiting_time()
                    n_pick_ups = 0
                    adv_reward = -waiting_time
                    tracker.track_adv_reward(adv_reward)
                    adv_memory.append((adv_inputs, assignment, adv_reward))

                if n_drop_offs == 2:
                    tracker.reset_waiting_time()
                    n_drop_offs = 0
                    adv_inputs = create_adv_inputs(next_states)
                    assignments = []
                    for adv_policy in adv_policies:
                        assignments.append(adv_policy(adv_inputs))

                    # check if agents agree on assignment
                    if cfg['decentralized']:
                        if assignments.count(assignments[0]) == len(assignments):
                            equal_assignments += 1
                        else:
                            different_assignments += 1

                    # randomly choose assignment
                    assignment = random.sample(assignments, 1)[0]
                    msgs = [0, 1] if assignment == 0 else [1, 0]
                    next_states = add_msg_to_states(next_states, msgs)
                else:
                    next_states = add_old_assignment(next_states, states)


            if episode >= cfg['adv_eps_without_training']:
                if steps % cfg['adv_update_freq'] == 0:
                    for adv in advs:
                        adv.replay(adv_memory, cfg['adv_replay_size'])

            if is_done:
                print(
                    f"Episode: {episode} \t Reward: {tracker.get_rewards()} \t Passengers {tracker.get_pick_ups()} ADV-Rewards {tracker.adv_episode_reward_arr[-1]:.2f}"
                )
                if cfg['decentralized']:
                    consens_ratio = round(
                        equal_assignments / max(equal_assignments + different_assignments, 1), 3)
                    consens_arr.append(consens_ratio)
                break

            states = next_states

        if episode >= cfg['adv_eps_without_training']:
            cfg['adv_epsilon'] = max(
                cfg['adv_epsilon'] * cfg['adv_epsilon_decay'], cfg['adv_epsilon_min'])

    for i, adv in enumerate(advs):
        adv.save_model(log_path, number=str(i + 1))

    tracker.plot(log_path)
    tracker.plot_adv_rewards(log_path)
    if cfg['decentralized']:
        tracker.plot_arr(consens_arr, log_path, "consens_ratio.png")
