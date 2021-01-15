import argparse

from algorithms.ppo import deploy_ppo

parser = argparse.ArgumentParser(description="Train: select algorithm and number of episodes")
parser.add_argument('-a', '--algorithm', type=str, required=True,
                    help="Algorithm to run")
parser.add_argument('-n', '--number', required=True,
                    help="Number of episodes to run")
args = parser.parse_args()

valid_algorithms = ['ppo', 'dqn']

if args.algorithm == "ppo": 
    deploy_ppo(int(args.number))
elif args.algorithms == 'dqn': 
    train_dqn(args.number)
else: 
    print('Not a valid algorithm: {args.algorithm}')

