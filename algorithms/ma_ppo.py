import os
import time
import numpy as np
from pyvirtualdisplay import Display

import gym
import gym_cabworld

from algorithms.ma_ppo_models import Memory, MAPPO
from common.features import clip_state, cut_off_state
from common.logging import create_log_folder, get_last_folder
from common.logging import MultiTracker


def train_ma_ppo(n_episodes):

    disp = Display()
    disp.start()

    env_name = "Cabworld-v1"
    env = gym.make(env_name)

    n_states = env.observation_space.shape[1]
    n_actions = env.action_space.n
    max_timesteps = 1000

    log_path = create_log_folder("mappo")
    tracker = MultiTracker(n_agents=2)

    memory = Memory()
    mappo = MAPPO(n_states, n_actions)

    for episode in range(n_episodes):

        tracker.new_episode()
        states = env.reset()

        total_reward = 0

        for _ in range(max_timesteps):

            action = mappo.policy_old.act(states, memory)
            states, rewards, done, _ = env.step(action)
            tracker.track_reward(rewards)

            total_reward += sum(rewards)

            memory.rewards.append(rewards)
            memory.is_terminal.append(done)

            if done:
                mappo.update(memory, episode)
                memory.clear()
                print(
                    f"Episode: {episode} Reward: {tracker.get_rewards()} Passengers {tracker.get_pick_ups()}"
                )
                break

    tracker.plot(log_path)
    mappo.save_model(log_path)


def deploy_ma_ppo(n_episodes, wait):

    env_name = "Cabworld-v1"
    env = gym.make(env_name)

    n_states = env.observation_space.shape[1]
    n_actions = env.action_space.n

    ppo = MAPPO(n_state=n_states, n_actions=n_actions)

    current_folder = get_last_folder("mappo")
    if not current_folder:
        print("No model")
        return
    current_model = os.path.join(current_folder, "mappo.pth")
    print(current_model)
    ppo.load_model(current_model)

    for _ in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = ppo.policy.deploy(state)
            state, reward, done, _ = env.step(action)
            episode_reward += sum(reward)
            env.render()
            time.sleep(float(wait))
            if done:
                print(f"Reward {gi}")
                break
