import os
import time
import numpy as np

import gym
import gym_cabworld

from algorithms.ppo_models import Memory, PPO
from common.features import clip_state, cut_off_state
from common.logging import create_log_folder, get_last_folder, create_logger
from common.logging import Tracker

from common.plotting import plot_losses

def train_ppo(n_episodes):

    from pyvirtualdisplay import Display

    Display().start()

    env_name = "Cabworld-v0"
    env = gym.make(env_name)

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    max_timesteps = 1000

    log_path = create_log_folder("ppo")
    logger = create_logger(log_path)
    tracker = Tracker(logger)

    memory = Memory()
    ppo = PPO(n_states, n_actions)

    trained_episodes = 0

    for episode in range(n_episodes):

        tracker.new_episode()
        state = env.reset()

        for _ in range(max_timesteps):

            action = ppo.policy_old.act(state, memory)
            old_state = state
            state, reward, done, _ = env.step(action)
            
            tracker.track_reward_and_action(reward, action, old_state)
            
            memory.rewards.append(reward)
            memory.is_terminal.append(done)

            if done:
                if tracker.get_pick_ups() > 0:
                    ppo.update(memory, episode)
                    trained_episodes += 1
                memory.clear()
                print(
                    f"Episode: {episode} Reward: {round(tracker.episode_reward,3)} Passengers {tracker.get_pick_ups()}"
                )
                break
    
    print("Trained episodes: ", trained_episodes)
    ppo.save_model(log_path)
    plot_losses(ppo.losses)
    tracker.plot(log_path)


def deploy_ppo(n_episodes, wait):

    env_name = "Cabworld-v0"
    env = gym.make(env_name)

    n_states = env.observation_space.shape[0]
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
