import numpy as np
import Grid

class AdvancedGrid(Grid):
    def __init__(self, transition_matrix, dimensions, data, random=0, living_reward=0, ):
        super().__init__(dimensions, data, random, living_reward)
        # A (x, y, d, (p, s', r)) matrix. 
        # d is max degree of actions, p specifies transition probability, s' is next_state, r is reward given.
        self.transition_matrix = transition_matrix 
        self.d = transition_matrix.shape[2]

    # to-do: implement reward and transition functions.