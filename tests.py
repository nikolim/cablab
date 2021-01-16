from algorithms.ppo import train_ppo, deploy_ppo
from algorithms.dqn import train_dqn, deploy_dqn
from algorithms.a2c import train_a2c, deploy_a2c
from algorithms.mdqn import train_mdqn, deploy_mdqn
from algorithms.random_policy import train_random, deploy_random

from common.features import clip_state, cut_off_state

def test_ppo_train():
    train_ppo(1)

def test_ppo_deploy():
    deploy_ppo(n_episodes=1, wait=0)

def test_dqn_train():
    train_dqn(1)

def test_dqn_deploy():
    deploy_dqn(n_episodes=1, wait=0)

def test_a2c_train():
    train_a2c(1)

def test_a2c_deploy():
    deploy_a2c(n_episodes=1, wait=0)

def test_mdqn_train():
    train_mdqn(1)

def test_mdqn_deploy():
    deploy_mdqn(n_episodes=1, wait=0)

def test_random_train():
    train_random(1)

def test_random_deploy():
    deploy_random(n_episodes=1, wait=0)

def test_clip_state(): 
    state = [1,2,3,4,5,6,7,8,9,10]
    new_state = clip_state(state, 4)
    for i in range(4,len(state)):
        assert new_state[i] == 0

def test_cut_off_state():
    state = [1,2,3,4,5,6,7,8,9,10]
    new_state = cut_off_state(state, 4)
    assert len(new_state) == 4