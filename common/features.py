import math
import random

def clip_state(state, n):
    clipped_state = (list(state))[:n]
    missing_pos = [0] * (len(state) - n)
    clipped_state += missing_pos
    return tuple(clipped_state)


def cut_off_state(state, n):
    state = list(state)
    state = state[:n]
    return tuple(state)


def passenger_potential(state, next_state):
    # increased potential for pick up or drop off
    potential = 1 if state[4] != next_state[4] else 0 
    return potential

def calc_add_signal(state):

    # calc distance cab to nearest passenger
    x_delta = state[6] - state[8]
    y_delta = state[7] - state[9]
    distance = math.sqrt((x_delta)**2 + (y_delta)**2) / math.sqrt(2)
    return 1 - distance

def calc_potential(state, next_state, gamma): 

    pot_state = calc_add_signal(state)
    pot_next_state = calc_add_signal(next_state)

    state_potential =  gamma * pot_next_state - pot_state

    return state_potential * 10
