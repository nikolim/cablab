import argparse

from algorithms.ppo import deploy_ppo
from algorithms.dqn import deploy_dqn
from algorithms.a2c import deploy_a2c
from algorithms.mdqn import deploy_mdqn
from algorithms.random_policy import deploy_random

parser = argparse.ArgumentParser(
    description="Train: select algorithm and number of episodes"
)
parser.add_argument(
    "-a", "--algorithm", type=str, required=True, help="Algorithm to run"
)
parser.add_argument("-n", "--number", required=True, help="Number of episodes to run")
parser.add_argument(
    "-w",
    "--wait",
    required=False,
    type=float,
    default=0.05,
    help="Delay between actions",
)
args = parser.parse_args()

valid_algorithms = ["ppo", "dqn"]

if args.algorithm == "ppo":
    deploy_ppo(int(args.number), int(args.wait))
elif args.algorithm == "dqn":
    deploy_dqn(int(args.number), int(args.wait))
elif args.algorithm == "a2c":
    deploy_a2c(int(args.number), int(args.wait))
elif args.algorithm == "mdqn":
    deploy_mdqn(int(args.number), int(args.wait))
elif args.algorithm == "rand":
    deploy_random(int(args.number), int(args.wait))
else:
    print(f"Not a valid algorithm: {args.algorithm}")
