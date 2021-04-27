import os

from algorithms.ppo import train_ppo, deploy_ppo
from algorithms.dqn import train_dqn, deploy_dqn
from algorithms.ma_dqn import train_ma_dqn, deploy_ma_dqn


def test_ppo_train_v0():
    train_ppo(n_episodes=10, version="v0")

def test_ppo_deploy_v0():
    deploy_ppo(n_episodes=10, version="v0")

def test_dqn_train_v0():
    train_dqn(n_episodes=10, version="v0")

def test_dqn_deploy_v0():
    deploy_dqn(n_episodes=10, version="v0")

def test_dqn_train_v1():
    train_dqn(n_episodes=10, version="v1")

def test_dqn_deploy_v1():
    deploy_dqn(n_episodes=10, version="v1")

def test_ma_dqn_train_v2():
    train_ma_dqn(n_episodes=10, version="v2")

def test_ma_dqn_train_v3():
    train_ma_dqn(n_episodes=10, version="v2")