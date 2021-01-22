import argparse

from algorithms.ppo import train_ppo
from algorithms.dqn import train_dqn
from algorithms.a2c import train_a2c
from algorithms.mdqn import train_mdqn
from algorithms.random_policy import train_random

from algorithms.machin_dqn import train_machin_dqn
from algorithms.machin_ppo import train_machin_ppo

from algorithms.ma_dqn import train_ma_dqn
from algorithms.ma_ppo import train_ma_ppo


train_ma_ppo(10)
