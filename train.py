import argparse

from algorithms.ppo import train_ppo
from algorithms.dqn import train_dqn
from algorithms.a2c import train_a2c
from algorithms.mdqn import train_mdqn
from algorithms.random_policy import train_random

parser = argparse.ArgumentParser(
    description="Train: select algorithm and number of episodes"
)
parser.add_argument(
    "-a", "--algorithm", type=str, required=True, help="Algorithm to run"
)
parser.add_argument("-n", "--number", required=True, help="Number of episodes to run")
args = parser.parse_args()

valid_algorithms = ["ppo", "dqn"]

if args.algorithm == "ppo":
    train_ppo(int(args.number))
elif args.algorithm == "dqn":
    train_dqn(int(args.number))
elif args.algorithm == "a2c":
    train_a2c(int(args.number))
elif args.algorithm == "mdqn":
    train_mdqn(int(args.number))
elif args.algorithm == "rand":
    train_random(int(args.number))
else:
    print("Not a valid algorithm: {args.algorithm}")
