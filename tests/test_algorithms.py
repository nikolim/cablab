import os

from algorithms.ppo import train_ppo, deploy_ppo
from algorithms.dqn import train_dqn, deploy_dqn
from algorithms.ma_dqn import train_ma_dqn, deploy_ma_dqn


def test_ppo_train():
    train_ppo(n_episodes=1, version="v0")

def test_ppo_deploy():
    deploy_ppo(n_episodes=1, version="v0")

def test_dqn_train():
    train_dqn(n_episodes=1, version="v0")

def test_dqn_deploy():
    deploy_dqn(n_episodes=1, version="v0")

def test_ma_dqn_train():
    train_ma_dqn(n_episodes=1, version="v2")

def test_ma_dqn_deploy():
    deploy_ma_dqn(n_episodes=1, version="v2")
