import argparse

from algorithms.ppo import run_ppo

parser = argparse.ArgumentParser(description="Training selector")
parser.add_argument('-a', '--algorithm', type=str, required=True,
                    help="Algorithm to run")
parser.add_argument('-n', '--number', type=int, required=True,
                    help="Number of episodes to run")

args = parser.parse_args()
print(args)

if args.algorithm == "ppo": 
    run_ppo(int(args.number))