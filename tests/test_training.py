import pytest

from cablab.algorithms.ppo import train_ppo, deploy_ppo
from cablab.algorithms.dqn import train_dqn
from cablab.algorithms.ma_dqn import train_ma_dqn


def test_ppo_train_v0():
    train_ppo(n_episodes=10, version="v0")


def test_dqn_train_v0():
    train_dqn(n_episodes=10, version="v0")


def test_ppo_train_v1():
    train_ppo(n_episodes=10, version="v1")


def test_dqn_train_v1():
    train_dqn(n_episodes=10, version="v1")


def test_ma_dqn_train_v2():
    train_ma_dqn(n_episodes=10, version="v2")


def test_ma_dqn_train_v3():
    train_ma_dqn(n_episodes=10, version="v3")
