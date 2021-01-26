from machin.frame.algorithms import MADDPG
from machin.utils.logging import default_logger as logger
from copy import deepcopy
import torch as t
import torch.nn as nn

import gym
import gym_cabworld


# configurations
env = gym.make("Cabworld-v1")
env.discrete_action_input = False
observe_dim = 14
action_num = 6
max_episodes = 100
max_steps = 1000
agent_num = 2


# model definition
class ActorConcrete(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorConcrete, self).__init__()

        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_dim)

    def forward(self, state):
        a = t.relu(self.fc1(state))
        a = t.relu(self.fc2(a))
        a = t.softmax(self.fc3(a), dim=1)
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):

        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, state, action):
        state_action = t.cat([state, action], 1)
        q = t.relu(self.fc1(state_action))
        q = t.relu(self.fc2(q))
        q = self.fc3(q)
        return q


if __name__ == "__main__":

    from pyvirtualdisplay import Display
    Display().start()

    actor = ActorConcrete(observe_dim, action_num)
    critic = Critic(observe_dim * agent_num, action_num * agent_num)

    maddpg = MADDPG(
        [deepcopy(actor) for _ in range(agent_num)],
        [deepcopy(actor) for _ in range(agent_num)],
        [deepcopy(critic) for _ in range(agent_num)],
        [deepcopy(critic) for _ in range(agent_num)],
        [list(range(agent_num))] * agent_num,
        t.optim.Adam,
        nn.MSELoss(reduction="sum"),
    )

    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0

    while episode < max_episodes:
        episode += 1
        total_reward = 0
        terminal = False
        step = 0
        states = [
            t.tensor(st, dtype=t.float32).view(1, observe_dim) for st in env.reset()
        ]

        while not terminal and step <= max_steps:
            step += 1
            with t.no_grad():
                old_states = states
                # agent model inference
                results = maddpg.act_discrete_with_noise(
                    [{"state": st} for st in states]
                )
                actions = [int(r[0]) for r in results]
                action_probs = [r[1] for r in results]

                states, rewards, terminals, _ = env.step(actions)
                states = [
                    t.tensor(st, dtype=t.float32).view(1, observe_dim) for st in states
                ]

                total_reward += float(sum(rewards)) / agent_num

                maddpg.store_transitions(
                    [
                        {
                            "state": {"state": ost},
                            "action": {"action": act},
                            "next_state": {"state": st},
                            "reward": float(rew),
                            "terminal": step == max_steps,
                        }
                        for ost, act, st, rew in zip(
                            old_states, action_probs, states, rewards
                        )
                    ]
                )

        # update, update more if episode is longer, else less
        if episode > 10:
            for _ in range(step):
                maddpg.update()

        # show reward
        print("Episode {} total reward={:.2f}".format(episode, total_reward))
