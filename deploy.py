import argparse

from algorithms.ppo import deploy_ppo
from algorithms.dqn import deploy_dqn
from algorithms.a2c import deploy_a2c
from algorithms.random_policy import deploy_random
from algorithms.ma_dqn import deploy_ma_dqn

parser = argparse.ArgumentParser(
    description="Train: select algorithm and number of episodes"
)
parser.add_argument(
    "-a", "--algorithm", type=str, required=True, help="Algorithm to run"
)
parser.add_argument("-n", "--number", type=int,
                    required=True, help="Number of episodes to run")
parser.add_argument(
    "-env",
    "--environment",
    type=str,
    required=True,
    help="Select Environment Version",
)
parser.add_argument(
    "-w",
    "--wait",
    required=False,
    type=float,
    default=0.05,
    help="Delay between actions",
)
parser.add_argument(
    "-r",
    "--render",
    required=False,
    type=bool,
    default=False,
    help="Render deploy runs",
)
parser.add_argument(
    "-e",
    "--eval",
    required=False,
    type=bool,
    default=False,
    help="Run eval runs",
)

args = parser.parse_args()

valid_algorithms = ["ppo", "dqn", "ma-dqn"]

if args.algorithm == "ppo":
    deploy_ppo(int(args.number), float(args.wait))
elif args.algorithm == "dqn":
    deploy_dqn(args.number, args.environment,
               args.eval, args.render, args.wait)
elif args.algorithm == "a2c":
    deploy_a2c(int(args.number), float(args.wait))
elif args.algorithm == "rand":
    deploy_random(int(args.number), float(args.wait))
elif args.algorithm == "ma-dqn":
    deploy_ma_dqn(int(args.number), float(args.wait), comm=bool(
        args.communication), render=bool(args.render), eval=bool(args.eval))
else:
    print(f"Not a valid algorithm: {args.algorithm}")
