import math

def euclidean_distance(p1, p2): 
    return round(1-math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)/math.sqrt(2), 5)

def feature_engineering(state): 

    dist_pos_pass1 = euclidean_distance((state[6], state[7]), (state[8], state[9]))
    dist_dest_pas1 = euclidean_distance((state[6], state[7]), (state[10], state[11]))

    if state[10] == -1: 
        dist_pos_pass2 = -1
        dist_dest_pas2 = -1
    else:
        dist_pos_pass2 = euclidean_distance((state[6], state[7]), (state[12], state[13]))
        dist_dest_pas2 = euclidean_distance((state[6], state[7]), (state[14], state[15]))
    if state[14] == -1: 
        dist_pos_pass3 = -1
        dist_dest_pas3 = -1
    else:
        dist_pos_pass3 = euclidean_distance((state[6], state[7]), (state[16], state[17]))
        dist_dest_pas3 = euclidean_distance((state[6], state[7]), (state[18], state[19]))

    state = list(state)
    new_state = state[:8] + [dist_pos_pass1, dist_dest_pas1] + state[8:12] + [dist_pos_pass2, dist_dest_pas2] + state[12:16] + [dist_pos_pass3, dist_dest_pas3] + state[16:]
    return new_state
