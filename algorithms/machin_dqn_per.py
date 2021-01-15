from machin.frame.algorithms import DQNPer
from machin.utils.logging import default_logger as logger
import torch as t
import torch.nn as nn
import gym

import gym_cabworld
from features import clip_state

# configurations
env = gym.make("Cabworld-v0")
observe_dim = 14
action_num = 6
max_episodes = 300
max_steps = 1000
solved_reward = 5000
solved_repeat = 5
n_clip = 6

# model definition
class QNet(nn.Module):
    def __init__(self, state_dim, action_num):
        super(QNet, self).__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_num)

    def forward(self, state):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        return self.fc3(a)


if __name__ == "__main__":
    q_net = QNet(observe_dim, action_num)
    q_net_t = QNet(observe_dim, action_num)

    dqn_per = DQNPer(q_net, q_net_t,
                     t.optim.Adam,
                     nn.MSELoss(reduction='sum'))

    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0

    while episode < max_episodes:
        episode += 1
        total_reward = 0
        terminal = False
        step = 0
        state = clip_state(env.reset(), n_clip)
        state = t.tensor(state, dtype=t.float32).view(1, observe_dim)

        while not terminal and step <= max_steps:
            step += 1
            with t.no_grad():
                old_state = state
                # agent model inference
                action = dqn_per.act_discrete_with_noise(
                    {"state": old_state}
                )
                state, reward, terminal, _ = env.step(action.item())
                state = clip_state(state, n_clip)
                state = t.tensor(state, dtype=t.float32).view(1, observe_dim)
                total_reward += reward

                dqn_per.store_transition({
                    "state": {"state": old_state},
                    "action": {"action": action},
                    "next_state": {"state": state},
                    "reward": reward,
                    "terminal": terminal or step == max_steps
                })

        # update, update more if episode is longer, else less
        if episode > 100:
            for _ in range(step):
                dqn_per.update()

        # show reward
        smoothed_total_reward = (smoothed_total_reward * 0.9 +
                                 total_reward * 0.1)
        logger.info("Episode {} total reward={:.2f}"
                    .format(episode, smoothed_total_reward))

        if smoothed_total_reward > solved_reward:
            reward_fulfilled += 1
            if reward_fulfilled >= solved_repeat:
                logger.info("Environment solved!")
                exit(0)
        else:
            reward_fulfilled = 0