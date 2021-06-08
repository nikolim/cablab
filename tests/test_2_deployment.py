import pytest

from cablab.algorithms.ppo import deploy_ppo
from cablab.algorithms.dqn import deploy_dqn
from cablab.algorithms.ma_dqn import deploy_ma_dqn


def test_ppo_deploy_v1():
    deploy_ppo(n_episodes=10, version="v1")


def test_dqn_train_v1():
    deploy_dqn(n_episodes=10, version="v1")


def test_dqn_deploy_v3():
    deploy_ma_dqn(n_episodes=10, version="v3")
