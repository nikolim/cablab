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
    distance = math.sqrt((x_delta) ** 2 + (y_delta) ** 2) / math.sqrt(2)
    return 1 - distance


def calc_potential(state, next_state, gamma):

    pot_state = calc_add_signal(state)
    pot_next_state = calc_add_signal(next_state)

    state_potential = gamma * pot_next_state - pot_state

    return state_potential * 10


def calc_shorter_way(states):
    def calc_distance(state):
        return round(
            math.sqrt((state[5] - state[7]) ** 2 + (state[6] - state[8]) ** 2), 2
        )
    distances = [calc_distance(state) for state in states]
    one_hot = [1 if dist == min(distances) else 0 for dist in distances]

    # if distances are equal select one randomly
    while one_hot.count(1) > 1:
        idx = [index for index, value in enumerate(one_hot) if value == 1]
        rand_idx = random.sample(idx, 1)[0]
        one_hot[rand_idx] = 0

    return one_hot


def add_fixed_msg_to_states(states):
    new_states = []
    one_hot_msg = calc_shorter_way(states)

    for state, msg in zip(states, one_hot_msg[::-1]):
        state_arr = list(state)
        state_arr.append(msg)
        new_states.append(tuple(state_arr))
    return new_states


def send_pos_to_other_cab(states):

    assert len(states) == 2

    # passenger is the same for all cabs
    pass_x, pass_y = states[0][7], states[0][8]

    cab1_pos_x, cab1_pos_y = states[0][5], states[0][6]
    cab2_pos_x, cab2_pos_y = states[1][5], states[1][6]

    adv1 = [cab1_pos_x, cab1_pos_y, pass_x, pass_y, cab2_pos_x, cab1_pos_y]
    adv2 = [cab2_pos_x, cab2_pos_y, pass_x, pass_y, cab1_pos_x, cab1_pos_y]

    return [adv1, adv2]


def add_msg_to_states(states, msgs):

    new_states = []

    for state, msg in zip(states, msgs[::-1]):
        state_arr = list(state)
        state_arr.append(msg)
        new_states.append(tuple(state_arr))

    return new_states


def calc_adv_rewards(adv_inputs, msgs):

    rewards = []

    for adv_input, msg in zip(adv_inputs, msgs):

        my_distance = round(
            math.sqrt(
                (adv_input[0] - adv_input[2]) ** 2 + (adv_input[1] - adv_input[3]) ** 2
            ),
            2,
        )
        other_distance = round(
            math.sqrt(
                (adv_input[2] - adv_input[4]) ** 2 + (adv_input[3] - adv_input[5]) ** 2
            ),
            2,
        )

        delta = my_distance - other_distance

        if delta > 0 and msg == 0:
            reward = 1
        elif delta <= 0 and msg == 0:
            reward = -1
        elif delta > 0 and msg == 1:
            reward = -1
        elif delta <= 0 and msg == 1:
            reward = 1

        rewards.append(reward)

    return rewards


def extend_single_agent_state(state):
    extended_state = list(state)
    if (state[4] == -1 and state[7] != -1 and state[8] != -1):
        extended_state.append(1)
    else:
        extended_state.append(0)
    return tuple(extended_state)


def assign_passenger(state):
    extended_state = list(state)
    extended_state.append(random.randint(0,1))
    return tuple(extended_state)

def picked_up_assigned_psng(state): 
    state = list(state)
    if state[5] == state[7] and state[6] == state[8]: 
        return True if state[-1] == 0 else False
    elif state[5] == state[9] and state[6] == state[10]: 
        return True if state[-1] == 1 else False
    else: 
        raise Exception("No-pick-up-possible")