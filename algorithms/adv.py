import os
import time
import json
import shutil
import random
import numpy as np

from collections import deque
from pyvirtualdisplay import Display

import gym
import gym_cabworld

from algorithms.adv_model import *
from algorithms.dqn_model import *
from common.features import *
from common.logging import *

cfg_path = "configs"
madqn_cfg_file = "ma_dqn_conf.json"
adv_cfg_file = "adv_conf.json"
madqn_cfg_file_path = os.path.join(cfg_path, madqn_cfg_file)
adv_cfg_file_path = os.path.join(cfg_path, adv_cfg_file)


def train_adv(n_episodes, version):

    assert version == "v3"

    # Start virtual display
    disp = Display()
    disp.start()

    cfg = json.load(open(madqn_cfg_file_path))
    print(f'Config loaded: {madqn_cfg_file_path}')
    adv_cfg = json.load(open(adv_cfg_file_path))
    print(f'Config loaded: {adv_cfg_file_path}')

    env_name = 'Cabworld-' + version
    env = gym.make(env_name)

    n_agents = env.observation_space.shape[0]
    n_states = env.observation_space.shape[1] + 1
    n_actions = env.action_space.n

    # If single or multiple adv-policies should be trained
    n_policies = n_agents if adv_cfg['decentralized'] else 1

    log_path = create_log_folder("adv")
    tracker = MultiTracker(n_agents=n_agents, version=version)

    # copy config-file into log-folder
    shutil.copyfile(adv_cfg_file_path,
                    os.path.join(log_path, adv_cfg_file))

    dqn = DQN(n_states, n_actions, cfg)

    # Select model trained on assigned passengers (single or multi-agent)
    current_model = "/home/niko/Info/cablab/runs/dqn/40/dqn.pth"
    dqn.load_model(current_model)

    advs = []
    for _ in range(n_policies):
        advs.append(AdvNet(n_input=n_agents*4, n_msg=n_agents, lr=adv_cfg['adv_lr']))

    adv_memory = deque(maxlen=adv_cfg['adv_memory_size'])
    consens_arr = []

    for episode in range(n_episodes + adv_cfg['adv_eps_without_training']):

        tracker.new_episode()
        tracker.reset_waiting_time()

        # temp episode counter
        n_pick_ups = 0
        n_drop_offs = 0
        equal_assignments = 0
        different_assignments = 0

        adv_policies = []
        for adv in advs:
            adv_policies.append(gen_epsilon_greedy_policy(
                adv, adv_cfg['adv_epsilon'], adv_cfg['n_msg']))

        states = env.reset()

        adv_inputs = create_adv_inputs(states)
        assignments = []
        for adv_policy in adv_policies:
            assignments.append(adv_policy(adv_inputs))

        # check if agents agree on assignment
        if adv_cfg['decentralized']:
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

            if version == 'v3':
                for reward, action in zip(rewards, actions):
                    if reward == 1:
                        if action == 4:
                            n_pick_ups += 1
                        else:
                            n_drop_offs += 1

                if n_pick_ups == 2:
                    waiting_time = tracker.reset_waiting_time()
                    n_pick_ups = 0
                    adv_reward = -waiting_time
                    tracker.track_adv_reward(adv_reward)
                    adv_memory.append((adv_inputs, assignment, adv_reward))

                if n_drop_offs == 2:
                    tracker.reset_waiting_time()
                    n_drop_offs = 0
                    assignments = []

                    for adv_policy in adv_policies:
                        assignments.append(adv_policy(adv_inputs))

                    # check if agents agree on assignment
                    if adv_cfg['decentralized']:
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

            tracker.track_reward_and_action(rewards, actions, states)

            if cfg['common_reward']:
                summed_rewards = sum(rewards)
                rewards = [summed_rewards for _ in rewards]

            if episode >= adv_cfg['adv_eps_without_training']:
                if steps % adv_cfg['update_freq'] == 0:
                    for adv in advs:
                        adv.replay(adv_memory, adv_cfg['adv_replay_size'])

            if is_done:
                print(
                    f"Episode: {episode} \t Reward: {tracker.get_rewards()} \t Passengers {tracker.get_pick_ups()} ADV-Rewards {tracker.adv_episode_reward_arr[-1]:.2f}"
                )
                if adv_cfg['decentralized']:
                    consens_ratio = round(
                        equal_assignments / max(equal_assignments + different_assignments, 1), 3)
                    consens_arr.append(consens_ratio)
                break

            states = next_states

        if episode >= adv_cfg['adv_eps_without_training']:
            adv_cfg['adv_epsilon'] = max(
                adv_cfg['adv_epsilon'] * adv_cfg['adv_epsilon_decay'], adv_cfg['adv_epsilon_min'])

    for i, adv in enumerate(advs):
        adv.save_model(log_path, number=str(i + 1))

    tracker.plot(log_path)
    tracker.plot_adv_rewards(log_path)
    if adv_cfg['decentralized']:
        tracker.plot_arr(consens_arr, log_path, "consens_ratio.png")
