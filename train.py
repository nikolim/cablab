import argparse

from algorithms.ppo import train_ppo
from algorithms.dqn import train_dqn
from algorithms.a2c import train_a2c
from algorithms.random_policy import train_random
from algorithms.ma_dqn import train_ma_dqn
from algorithms.ma_ppo import train_ma_ppo


parser = argparse.ArgumentParser(
    description="Train: select algorithm and number of episodes"
)
parser.add_argument(
    "-a", "--algorithm", type=str, required=True, help="Algorithm to run"
)
parser.add_argument("-n", "--number", required=True, help="Number of episodes to run")
parser.add_argument(
    "-m",
    "--munchhausen",
    type=bool,
    default=False,
    required=False,
    help="Munchhausen add-on for DQN",
)

parser.add_argument(
    "-adv",
    "--advice",
    type=bool,
    default=False,
    required=False,
    help="AdvNet for  MA-DQN",
)

parser.add_argument(
    "-comm",
    "--communication",
    type=bool,
    default=False,
    required=False,
    help="Communication for  MA-DQN",
)


args = parser.parse_args()

valid_algorithms = ["ppo", "dqn", "a2c", "rand"]

if args.algorithm == "ppo":
    train_ppo(int(args.number))
if args.algorithm == "dqn":
    train_dqn(int(args.number), munchhausen=args.munchhausen)
elif args.algorithm == "a2c":
    train_a2c(int(args.number))
elif args.algorithm == "rand":
   train_random(int(args.number))
elif args.algorithm == "ma-dqn":
    train_ma_dqn(
        int(args.number),
        munchhausen=args.munchhausen,
        adv=args.advice,
        comm=args.communication,
    )
elif args.algorithm == "ma-ppo":
    train_ma_ppo(int(args.number))
else:
    print("Not a valid algorithm: {args.algorithm}")
