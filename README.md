<div align="center">
		<img width="auto" height="200px" src="img/icon.png">
</div>

<br/>
<div align="center">
	<a href="https://opensource.org/licenses/MIT">
		<img alt="License MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg">
	</a>
	<a href="https://opensource.org/licenses/MIT">
		<img alt="coverage" src="https://img.shields.io/badge/coverage-90%25-green">
	</a>
</div>

# Cablab

Development, improvement and testing of various reinforcement learning algorithms. <br>
Focus is on finding an effective algorithm for the cabworld. The algorithms can also be applied to all other OpenAIGym environments.

## Cabworld

In the cabworld a cab is driving around and tries to pick-up passengers to drive them to their destination.
Each passenger is marked with the same color as his destination. <br>
https://gitlab.com/nlimbrun/cabworld

## Currently implemented Algorithms

1. DQN
5. MDQN
2. A2C
3. PPO
4. MA-PPO (individual)

## Subtasks​

The cabworld is a relativly complex environment with different subtasks which can be seperated

1. Subtask 1: Learn movements 
2. Subtask 2: Learn pick-ups 
3. Subtask 3: Learn to select passengers 

## Requirements 

- Pytorch 
- Gym-Cabworld 

## Usage

### Training
```bash
python3 train.py -a ppo -n 1
```
```
usage: python3 train.py -a ALGORITHM -n NUMBER
error: the following arguments are required: -a/--algorithm, -n/--number
```

### Deploy 

Uses the most recent model

```bash
python3 deploy.py -a ppo -n 1 -w 0.05
```
```
usage: python3 deploy.py -a ALGORITHM -n NUMBER [-w WAIT]
error: the following arguments are required: -a/--algorithm, -n/--number
```


## Changelog

### [0.2] (https://gitlab.com/nlimbrun/cablab/-/tags/release_0.2) (16.01.2021)
- Unified training and deployment, add CI

### [0.1] (https://gitlab.com/nlimbrun/cablab/-/tags/release_0.1) (10.01.2021)
- Extended README