import os
import time
import json
import shutil
import numpy as np

from collections import deque
from pyvirtualdisplay import Display

import gym
import gym_cabworld

from algorithms.advnet import *
from algorithms.dqn_model import *
from common.features import *
from common.logging import *

cfg_path = "configs"
madqn_cfg_file = "ma_dqn_conf.json"
madqn_cfg_file_path = os.path.join(cfg_path, madqn_cfg_file)


def train_ma_dqn(n_episodes, version):

    # Start virtual display
    disp = Display()
    disp.start()

    # Load configuration
    cfg = json.load(open(madqn_cfg_file_path))
    # make sure to only select one option to extend state
    assert [cfg['info'],cfg['adv'], cfg['assign_psng']].count(True) <= 1
    cfg['episodes_only_adv'] = cfg['episodes_only_adv'] if cfg['adv'] else 0

    env_name = 'Cabworld-' + version
    env = gym.make(env_name)

    if cfg['adv'] or cfg['assign_psng']:
        extended_state = 1
    elif cfg['info']:
        extended_state = 2
    else:
        extended_state = 0

    n_agents = env.observation_space.shape[0]
    n_states = env.observation_space.shape[1] + extended_state
    n_actions = env.action_space.n

    log_path = create_log_folder("ma-dqn")
    tracker = MultiTracker(n_agents=n_agents, version=version)

    # copy config-file into log-folder
    shutil.copyfile(madqn_cfg_file_path,
                    os.path.join(log_path, madqn_cfg_file))

    dqn = DQN(n_states, n_actions, cfg)
    memory = deque(maxlen=cfg['episodes_without_training']
                   * env.spec.max_episode_steps)

    if cfg['adv']:
        adv = AdvNet(n_input=n_agents*4, n_msg=n_agents)
        adv_memory = deque(maxlen=cfg['adv_memory_size'])

    for episode in range(n_episodes + cfg['episodes_without_training'] + cfg['episodes_only_adv']):

        tracker.new_episode()

        if episode % cfg['target_update'] == 0:
            dqn.copy_target()

        policy = gen_epsilon_greedy_policy(
            dqn, cfg['epsilon'], n_actions)

        if cfg['adv']:
            adv_policy = gen_epsilon_greedy_policy(
                adv, cfg['adv_epsilon'], cfg['n_msg']
            )

        states = env.reset()

        if cfg['info']:
            # Stage 1 -> append positon of other agents
            states = append_other_agents_pos(states)
        if cfg['adv']:
            # Stage 2 -> give active advice
            adv_inputs = create_adv_inputs(states)
            assignment = adv_policy(adv_inputs)
            msgs = [0, 1] if assignment == 0 else [1, 0]
            states = add_msg_to_states(states, msgs)
        if cfg['assign_psng']: 
            msgs = optimal_assignment(states)
            states = add_msg_to_states(states, msgs)

        is_done = False
        steps = 0

         # temp pick-up and drop-off counter for v3
        n_pick_ups = 0
        n_drop_offs = 0

        while not is_done:

            steps += 1
            actions = []
            for i in range(n_agents):
                actions.append(policy(states[i]))

            next_states, rewards, is_done, _ = env.step(actions)

            if cfg['info']:
                next_states = append_other_agents_pos(next_states)

            if version == 'v3':
                for reward, action in zip(rewards, actions):
                    if reward == 1:
                        if action == 4:
                            n_pick_ups += 1
                        else:
                            n_drop_offs += 1
                if n_pick_ups == 2:
                    waiting_time = tracker.reset_waiting_time(log=True)
                    n_pick_ups = 0
                    if cfg['adv']:
                        adv_reward = -(waiting_time / 1000)
                        tracker.track_adv_reward(adv_reward)
                        adv_memory.append((adv_inputs, assignment, adv_reward))

                if n_drop_offs == 2:
                    tracker.reset_waiting_time(log=True)
                    n_drop_offs = 0
                    if cfg['adv']:
                        adv_inputs = create_adv_inputs(next_states)
                        assignment = adv_policy(adv_inputs)
                        msgs = [0, 1] if assignment == 0 else [1, 0]
                        next_states = add_msg_to_states(next_states, msgs)
                    if cfg['assign_psng']:
                        msgs = optimal_assignment(next_states)
                        next_states = add_msg_to_states(next_states, msgs)
        
                else:
                    if cfg['adv'] or cfg['assign_psng']:
                        next_states = add_old_assignment(next_states, states)

            tracker.track_reward_and_action(rewards, actions, states)

            if version == 'v3' and (cfg['adv'] or cfg['assign_psng']):
                # scale rewards for pick-up if passenger was assigned
                for reward, action, state, i in zip(rewards, actions, states, list(range(n_agents))):
                    if action == 4 and reward == 1:
                        if picked_up_assigned_psng(state):
                            rewards[i] = rewards[i] * cfg['assign_factor']
                        else:
                            rewards[i] = 0 # rewards[i] / cfg['assign_factor']

            if cfg['common_reward']:
                summed_rewards = sum(rewards)
                rewards = [summed_rewards for _ in rewards]

            for i in range(n_agents):
                memory.append(
                    (states[i], actions[i],
                     next_states[i], rewards[i], is_done)
                )

                if episode > cfg['episodes_without_training'] and steps % cfg['update_freq'] == 0:

                    if episode > (cfg['episodes_without_training'] + n_episodes):
                        adv.replay(adv_memory, cfg['adv_replay_size'])

                    if episode < (cfg['episodes_without_training'] + n_episodes):
                        if cfg['munchhausen']:
                            dqn.replay_munchhausen(
                                memory, cfg['replay_size'], cfg['gamma']
                            )
                        else:
                            dqn.replay(
                                memory, cfg['replay_size'], cfg['gamma'])

            if is_done:
                adv_rewards = f"ADV {tracker.adv_episode_rewards}" if cfg['adv'] else ""
                print(
                    f"Episode: {episode} \t Reward: {tracker.get_rewards()} \t Passengers {tracker.get_pick_ups()}"
                )
                # selective pops
                if sum(tracker.get_pick_ups()) < cfg['min_pick_ups']:
                    for _ in range(n_agents * env.spec.max_episode_steps):
                        memory.pop()
                break

            states = next_states

        if episode > cfg['episodes_without_training']:
            cfg['adv_epsilon'] = max(
                cfg['adv_epsilon'] * cfg['adv_epsilon_decay'], 0.01)

        if episode > (cfg['episodes_without_training']):
            cfg['epsilon'] = max(
                cfg['epsilon'] * cfg['epsilon_decay'], cfg['epsilon_min'])

    for i in range(n_agents):
        dqn.save_model(log_path, number=str(i + 1))
        if cfg['adv']:
            adv.save_model(log_path, number=str(i + 1))

    tracker.plot(log_path)
    if cfg['adv']:
        tracker.plot_adv_rewards(log_path)


def deploy_ma_dqn(n_episodes, version, eval=False, render=False, wait=0.05):

    # load config
    current_folder = get_last_folder("ma-dqn")
    cfg_file_path = os.path.join(current_folder, madqn_cfg_file)
    cfg = json.load(open(cfg_file_path))
    print(f'Config loaded: {cfg_file_path}')

    env_name = "Cabworld-" + version
    env = gym.make(env_name)

    if not render:
        disp = Display()
        disp.start()

    if cfg['adv'] or cfg['assign_psng']:
        extended_state = 1
    elif cfg['info']:
        extended_state = 2
    else:
        extended_state = 0

    n_agents = env.observation_space.shape[0]
    n_states = env.observation_space.shape[1] + extended_state
    n_actions = env.action_space.n

    tracker = MultiTracker(n_agents, version)

    # load model
    dqn = DQN(n_states, n_actions, cfg)
    current_model = os.path.join(current_folder, "dqn1.pth")
    print(current_model)
    dqn.load_model(current_model)

    if cfg['adv']:
        adv = AdvNet()
        current_adv_model = os.path.join(
            current_folder, "adv1.pth"
        )
        adv.load_model(current_adv_model)

    for episode in range(n_episodes):
        tracker.new_episode()
        states = env.reset()

        if cfg['info']:
            # Stage 1 -> append positon of other agents
            states = append_other_agents_pos(states)
        elif cfg['adv']:
            adv_inputs = create_adv_inputs(states)
            msgs = []
            for i in range(n_agents):
                msgs.append(adv.deploy((adv_inputs[i])))
            states = add_msg_to_states(states, msgs)
        elif cfg['assign_psng']:
            states = optimal_assignment(states)

        # temp pick up and drop off counter for v3
        n_pick_ups = 0
        n_drop_offs = 0

        done = False

        while not done:
            actions = [dqn.deploy(state)
                       for state in (states)]
            old_states = states
            states, rewards, done, _ = env.step(actions)

            if cfg['info']:
                # Stage 1 -> append positon of other agents
                states = append_other_agents_pos(states)

            if version == 'v3':
                for reward, action in zip(rewards, actions):
                    if reward == 1:
                        if action == 4:
                            n_pick_ups += 1
                        else:
                            n_drop_offs += 1

                if n_pick_ups == 2:
                    tracker.reset_waiting_time()
                    n_pick_ups = 0

                if n_drop_offs == 2:
                    tracker.reset_waiting_time(log=False)
                    n_drop_offs = 0
                    if cfg['adv']:
                        adv_inputs = create_adv_inputs(next_states)
                        assignment = adv[0].deploy(adv_inputs)
                        msgs = [0, 1] if assignment == 0 else [1, 0]
                        next_states = add_msg_to_states(next_states, msgs)
                    else:
                        states = add_old_assignment(states, old_states)

            tracker.track_reward_and_action(rewards, actions, states)

            if render:
                env.render()
                time.sleep(wait)
            if done:
                print(
                    f"Episode: {episode} \t Reward: {tracker.get_rewards()} \t Passengers {tracker.get_pick_ups()}"
                )
                break

    if eval:
        tracker.plot(log_path=current_folder, eval=True)


def train_only_adv(n_episodes, version):

    assert version == "v3"

    # Start virtual display
    disp = Display()
    disp.start()

    # load config
    current_folder = "/home/niko/Info/cablab/runs/ma-dqn/14"
    cfg_file_path = os.path.join(current_folder, madqn_cfg_file)
    cfg = json.load(open(cfg_file_path))

    env_name = 'Cabworld-' + version
    env = gym.make(env_name)

    n_agents = env.observation_space.shape[0]
    n_states = env.observation_space.shape[1] + 1
    n_actions = env.action_space.n

    log_path = create_log_folder("adv")
    tracker = MultiTracker(n_agents=n_agents, version=version)

    dqn = DQN(n_states, n_actions, cfg)
    current_model = os.path.join(current_folder, "dqn1.pth")
    dqn.load_model(current_model)

    adv = AdvNet(n_input=n_agents*4, n_msg=n_agents)
    adv_memory = deque(maxlen=cfg['adv_memory_size'])

    for episode in range(n_episodes):

        tracker.new_episode()
        tracker.reset_waiting_time(log=False)

        # temp pick-up and drop-off counter
        n_pick_ups = 0
        n_drop_offs = 0

        adv_policy = gen_epsilon_greedy_policy(
            adv, cfg['adv_epsilon'], cfg['n_msg'])

        states = env.reset()

        # Stage 2 -> give active advice
        adv_inputs = create_adv_inputs(states)
        assignment = adv_policy(adv_inputs)
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
                    adv_reward = -(waiting_time / 100)
                    tracker.track_adv_reward(adv_reward)
                    adv_memory.append((adv_inputs, assignment, adv_reward))

                if n_drop_offs == 2:
                    tracker.reset_waiting_time(log=False)
                    n_drop_offs = 0
                    adv_inputs = create_adv_inputs(next_states)
                    assignment = adv_policy(adv_inputs)
                    msgs = [0, 1] if assignment == 0 else [1, 0]
                    msgs = optimal_assignment_adv(adv_inputs)
                    # msgs.reverse()
                    next_states = add_msg_to_states(next_states, msgs)
                else:
                    next_states = add_old_assignment(next_states, states)

            tracker.track_reward_and_action(rewards, actions, states)

            if cfg['common_reward']:
                summed_rewards = sum(rewards)
                rewards = [summed_rewards for _ in rewards]

            for i in range(n_agents):

                if steps % cfg['update_freq'] == 0:
                    pass
                    #if episode > (cfg['episodes_without_training'] + n_episodes):
                    #    adv.replay(adv_memory, cfg['adv_replay_size'])

            if is_done:
                print(
                    f"Episode: {episode} \t Reward: {tracker.get_rewards()} \t Passengers {tracker.get_pick_ups()} ADV-Rewards {tracker.adv_episode_reward_arr[-1]}"
                )
                break

            states = next_states

        cfg['adv_epsilon'] = max(
            cfg['adv_epsilon'] * cfg['adv_epsilon_decay'], 0.01)

    for i in range(n_agents):
        adv.save_model(log_path, number=str(i + 1))

    tracker.plot(log_path)
    tracker.plot_adv_rewards(log_path)
