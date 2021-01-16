def clip_state(state, n):
    clipped_state = (list(state))[:n]
    missing_pos = [0] * (len(state) - n)
    clipped_state += missing_pos
    return tuple(clipped_state)


def cut_off_state(state, n):
    state = list(state)
    state = state[:n]
    return tuple(state)
