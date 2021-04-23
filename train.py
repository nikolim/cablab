import argparse

from algorithms.ppo import train_ppo
from algorithms.dqn import train_dqn
from algorithms.ma_dqn import train_ma_dqn

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
    "-r",
    "--runs",
    type=int,
    required=False,
    default=1,
    help="Select number of runs",
)

args = parser.parse_args()

if args.algorithm in ["ppo", "dqn"]:
    if args.environment not in ["v0", "v1"]:
        print(
            f"Error: {args.environment} is not a valid single agent environment")
        print(f"Please choose between: v0 and v1")
        quit()

elif args.algorithm in ["ma-dqn"]:
    if args.environment not in ["v2", "v3"]:
        print(f"{args.environment} is not a valid multi agent environment")
        print(f"Please choose between: v2 and v3")
        quit()
else:
    print(f"Error: {args.algorithm} is not a valid algorithm")
    print(f"Please choose between: dqn, ppo, ma-dqn")
    quit()

for _ in range(args.runs):
    if args.algorithm == "ppo":
        train_ppo(n_episodes=args.number, version=args.environment)
    if args.algorithm == "dqn":
        train_dqn(n_episodes=args.number, version=args.environment)
    elif args.algorithm == "ma-dqn":
        train_ma_dqn(
            n_episodes=args.number,
            version=args.environment)
