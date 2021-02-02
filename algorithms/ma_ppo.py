import os
import time
import numpy as np

from pyvirtualdisplay import Display
Display().start()

import gym
import gym_cabworld

from algorithms.ma_ppo_models import Memory, MAPPO
from common.features import clip_state, cut_off_state
from common.logging import create_log_folder, get_last_folder
from common.logging import MultiTracker


def train_ma_ppo(n_episodes):

    Display().start()
    env_name = "Cabworld-v1"
    env = gym.make(env_name)

    n_states = 14
    n_actions = 6
    max_timesteps = 1000

    log_path = create_log_folder("mappo")
    tracker = MultiTracker(n_agents=2)

    memory = Memory()
    mappo = MAPPO(n_states, n_actions)

    for episode in range(n_episodes):

        tracker.new_episode()
        states = env.reset()
        # state = clip_state(state, n_clip)
        # state = cut_off_state(state, n_state)

        total_reward = 0

        for _ in range(max_timesteps):

            action = mappo.policy_old.act(states, memory)
            states, rewards, done, _ = env.step(action)
            # state = clip_state(state, n_clip)
            # state = cut_off_state(state, n_state)

            tracker.track_reward(rewards)

            total_reward += sum(rewards)

            memory.rewards.append(rewards)
            memory.is_terminal.append(done)

            if done:
                mean_entropy = mappo.update(memory, episode)
                memory.clear()
                print(
                    f"Episode: {episode} Reward: {tracker.episode_reward} Passengers {tracker.get_pick_ups()}"
                )
                break

    # mappo.save_model(log_path)
    tracker.plot(log_path)


def deploy_ma_ppo(n_episodes, wait):

    env_name = "Cabworld-v1"
    env = gym.make(env_name)

    ppo = MAPPO(n_state=14, n_actions=6)

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
            episode_reward += reward
            env.render()
            time.sleep(float(wait))
            if done:
                print(f"Reward {episode_reward}")
                break
