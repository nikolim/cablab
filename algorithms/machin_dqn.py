from machin.frame.algorithms import DQN
from machin.utils.logging import default_logger as logger
import torch as t
import torch.nn as nn

import gym
import gym_cabworld

from common.logging import Tracker
from common.logging import create_log_folder


# configurations
env = gym.make("Cabworld-v0")
observe_dim = 14
action_num = 6
max_steps = 1000
solved_reward = 10000
solved_repeat = 5
n_clip = 6

log_path = create_log_folder("machin-dqn")
tracker = Tracker()

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


def train_machin_dqn(n_episodes):

    from pyvirtualdisplay import Display
    Display().start()

    q_net = QNet(observe_dim, action_num)
    q_net_t = QNet(observe_dim, action_num)

    dqn = DQN(
        q_net, q_net_t, t.optim.Adam, nn.MSELoss(reduction="sum"), replay_size=50000
    )

    episode, step = 0, 0
    total_reward = 0

    for episode in range(n_episodes):

        tracker.new_episode()

        episode += 1
        total_reward = 0
        terminal = False
        step = 0
        state = env.reset()
        state = t.tensor(state, dtype=t.float32).view(1, observe_dim)

        while not terminal and step <= max_steps:
            step += 1
            with t.no_grad():
                old_state = state
                # agent model inference
                action = dqn.act_discrete_with_noise({"state": old_state})
                state, reward, terminal, _ = env.step(action.item())
                state = t.tensor(state, dtype=t.float32).view(1, observe_dim)

                tracker.track_reward(reward)

                total_reward += reward

                dqn.store_transition(
                    {
                        "state": {"state": old_state},
                        "action": {"action": action},
                        "next_state": {"state": state},
                        "reward": reward,
                        "terminal": terminal or step == max_steps,
                    }
                )

        # update, update more if episode is longer, else less
        if episode > 50:
            for _ in range(step):
                dqn.update()

        print(
            f"Episode: {episode} Reward: {tracker.episode_reward} Passengers {tracker.get_pick_ups()}"
        )

    tracker.plot(log_path)
