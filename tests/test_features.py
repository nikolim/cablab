from random import randint, random

from common.features import *

def create_random_pos(): 
    return [round((1/7)*randint(0,7),3) for _ in range(2)]

def create_random_state(length): 

    assert length >= 9 # minimum state length for cabworld
    flags = [-1 if random() < 0.5 else 1 for _ in range(5)]
    positions = [round((1/7)*randint(0,7),3) for _ in range(length-len(flags))]

    return flags + positions

def test_calc_distance(): 
    calc_distance(create_random_pos(), create_random_pos())
