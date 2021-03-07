import os
import time
import numpy as np

import gym
import gym_cabworld

from algorithms.ppo_models import Memory, PPO
from common.features import clip_state, cut_off_state
from common.logging import create_log_folder, get_last_folder
from common.logging import Tracker


def train_ppo(n_episodes):

    from pyvirtualdisplay import Display

    Display().start()

    env_name = "Cabworld-v0"
    env = gym.make(env_name)

    n_states = env.observation_space.shape[1]
    n_actions = env.action_space.n
    max_timesteps = 1000

    log_path = create_log_folder("ppo")
    tracker = Tracker()

    memory = Memory()
    ppo = PPO(n_states, n_actions)

    for episode in range(n_episodes):

        tracker.new_episode()
        state = env.reset()

        for _ in range(max_timesteps):

            action = ppo.policy_old.act(state, memory)
            old_state = state
            state, reward, done, _ = env.step(action)
            tracker.track_reward(reward)

            memory.rewards.append(reward)
            memory.is_terminal.append(done)

            if done:
                ppo.update(memory, episode)
                memory.clear()
                print(
                    f"Episode: {episode} Reward: {tracker.episode_reward} Passengers {tracker.get_pick_ups()}"
                )
                break

    ppo.save_model(log_path)
    tracker.plot(log_path)


def deploy_ppo(n_episodes, wait):

    env_name = "Cabworld-v0"
    env = gym.make(env_name)

    n_states = env.observation_space.shape[1]
    n_actions = env.action_space.n

    ppo = PPO(n_state=n_states, n_actions=n_actions)

    current_folder = get_last_folder("ppo")
    if not current_folder:
        print("No model")
        return
    current_model = os.path.join(current_folder, "ppo.pth")
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
