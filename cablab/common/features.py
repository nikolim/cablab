import math
import random


def calc_distance(pos1, pos2):
    """
    Calculate euclidean distance between two positions 
    """
    return round(math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2), 3)


def create_adv_inputs(states):
    """
    Create the input for the ADV Net: cab-positions + psng-positions
    """
    # For testing this only works for 2 agents and 2 passengers
    assert len(states) == 2
    assert len(states[0]) == 11

    # passengers are the same for all cabs
    pass1_x, pass1_y = states[0][7], states[0][8]
    pass2_x, pass2_y = states[0][9], states[0][10]

    cab1_pos_x, cab1_pos_y = states[0][5], states[0][6]
    cab2_pos_x, cab2_pos_y = states[1][5], states[1][6]

    return [cab1_pos_x, cab1_pos_y, cab2_pos_x, cab2_pos_y, pass1_x, pass1_y, pass2_x, pass2_y]


def create_adv_inputs_single(states):
    """
    Create the input for the ADV Net: cab-positions + psng-positions
    """
    # For testing this only works for 2 agents and 2 passengers
    assert len(states) == 2
    assert len(states[0]) == 9

    # passenger is the same for all cabs
    pass1_x, pass1_y = states[0][7], states[0][8]

    cab1_pos_x, cab1_pos_y = states[0][5], states[0][6]
    cab2_pos_x, cab2_pos_y = states[1][5], states[1][6]

    return [cab1_pos_x, cab1_pos_y, cab2_pos_x, cab2_pos_y, pass1_x, pass1_y]


def add_msg_to_states(states, msgs):
    """
    Concate respective states with msgs
    """
    new_states = []
    for state, msg in zip(states, msgs):
        state_arr = list(state) + [msg]
        new_states.append(tuple(state_arr))
    return new_states


def assign_passenger(state):
    """
    Randomly assign passenger
    """
    extended_state = list(state)
    extended_state.append(random.randint(0, 1))
    return tuple(extended_state)


def single_agent_assignment(reward, action, state, next_state, tracker):
    """
    Randomly assign passenger after drop-off else keep old assignment
    """
    if reward == 1:
        if action == 4:
            if picked_up_assigned_psng(state):
                #tracker.assigned_psng += 1
                reward = 2
            else:
                #tracker.wrong_psng += 1
                reward = 0
        else:
            # assign new passenger after drop-off
            return assign_passenger(next_state), reward
    # keep old assignment
    return list(next_state) + [state[-1]], reward


def picked_up_assigned_psng(state):
    """
    Check if the agent picked up the assigned passenger v3
    Compare current postion of agent with the positions of the passengers
    """
    state = list(state)
    if round(state[5], 3) == round(state[7], 3) and round(state[6], 3) == round(state[8], 3):
        return True if state[-1] == 0 else False
    elif round(state[5], 3) == round(state[9], 3) and round(state[6], 3) == round(state[10], 3):
        return True if state[-1] == 1 else False
    else:
        raise Exception("No-pick-up-possible")
        # return random.sample([True, False], 1)[0]


def passenger_assigned(state):
    """
    Check if passenger was assinged v2
    """
    return state[-1] == 1


def random_assignment(states):
    """
    Append random but distinct assignments to states
    """
    a = random.sample([-1, 1], 1)[0]
    b = -1 if a == 1 else 1
    return add_msg_to_states(states, [a, b])


def optimal_assignment(states):
    """
    Calculate optimal assignment based on euclidean distance
    """
    a_1 = calc_distance((states[0][5], states[0][6]),
                        (states[0][7], states[0][8]))
    b_1 = calc_distance((states[1][5], states[1][6]),
                        (states[1][7], states[1][8]))

    a_2 = calc_distance((states[0][5], states[0][6]),
                        (states[0][9], states[0][10]))
    b_2 = calc_distance((states[1][5], states[1][6]),
                        (states[1][9], states[1][10]))

    assignment = [0, 1] if (a_1 + b_2) < (a_2 + b_1) else [1, 0]
    return add_msg_to_states(states, assignment)


def optimal_assignment_adv(adv_input):
    """
    Calculate optimal assignment based on euclidean distance
    """
    a_1 = calc_distance((adv_input[0], adv_input[1]),
                        (adv_input[4], adv_input[5]))
    b_1 = calc_distance((adv_input[2], adv_input[3]),
                        (adv_input[4], adv_input[5]))

    a_2 = calc_distance((adv_input[0], adv_input[1]),
                        (adv_input[6], adv_input[7]))
    b_2 = calc_distance((adv_input[2], adv_input[3]),
                        (adv_input[6], adv_input[7]))

    return ([0, 1] if (a_1 + b_2) < (a_2 + b_1) else [1, 0])


def add_old_assignment(next_states, states):
    """
    Keep assignment of previous state
    """
    new_states = []
    for next_state, state in zip(next_states, states):
        new_states.append(tuple((list(next_state)) + [state[-1]]))
    return new_states


def append_other_agents_pos(states):
    """
    Append the position of the other agent at the end of state
    """
    assert len(states) == 2
    new_states = []
    new_states.append(tuple(list(states[0]) + [states[1][5]] + [states[1][6]]))
    new_states.append(tuple(list(states[1]) + [states[0][5]] + [states[0][6]]))
    return new_states


def passenger_spawn(state, next_state):
    """
    Compare state with next state to determine if new passenger is spawn
    """
    return (state[7] == -1 and state[8] == -1) and (next_state[7] != -1 and next_state[8] != -1)
