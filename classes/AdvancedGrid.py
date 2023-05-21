import numpy as np
import Grid

class AdvancedGrid(Grid):
    def __init__(self, transition_matrix, dimensions, data, random=0, living_reward=0, ):
        super().__init__(dimensions, data, random, living_reward)
        # A (x, y, d, (p, s', r)) matrix. 
        # d is max degree of actions, p specifies transition probability, s' is next_state, r is reward given.
        self.transition_matrix = transition_matrix 
        self.d = transition_matrix.shape[2]
        self.simple = False

    # to-do: implement reward and transition functions.
    def transition(self, state: tuple, action: int):
        x = state[0]
        y = state[1]
        transition_vector = self.transition_matrix[x][y][action]
        ind, thresh, chance = -1, 0, np.random.random()
        while chance > thresh:
            ind += 1
            thresh += transition_vector[ind][0] # Get probability of some state
        return transition_vector[ind][1]
    
    def reward(self, state, action, next_state):
        if self.is_terminal(state):
            return 0
        
        for p, next, r in self.transition_matrix[state[0]][state[1]][action]:
            if next == next_state:
                return r + self.living_reward
        return self.living_reward
    
    def expected_reward(self, state: tuple, action: int):
        try:
            return self.expectation_matrix[state[0], state[1], action]
        except AttributeError:
            expect = 0
            for p, next, r in self.transition_matrix[state[0]][state[1]][action]:
                expect += p * r
            return expect
    
    def build_expectation_matrix(self):
        self.expectation_matrix = np.zeros((self.x, self.y, self.d))
        for x in range(self.x):
            for y in range(self.y):
                for action, tup in enumerate(self.transition_matrix[x][y]):
                    expect = 0
                    for p, next, r in tup:
                        expect += p * r