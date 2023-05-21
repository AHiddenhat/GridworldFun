import numpy as np
from classes.Agent import Agent

class ValueAgent(Agent):
    def __init__(self, grid, random_init=False, epsilon=1, gamma=0.9):
        super().__init__(grid, epsilon, gamma)
        if random_init: 
            self.value_matrix = np.random.rand(self.x, self.y)
        else:
            self.value_matrix = np.zeros(grid.dimensions)

    def get_action(self, state, eps=True) -> int:
        actions = self.grid.get_actions(state)
        if eps and np.random.random() < self.epsilon:
            return np.random.randint(0, len(actions))
        else:
            best, best_val = 0, np.NINF
            for action, _next in enumerate(actions):
                if self.grid.simple:
                    next_state = self.grid.transition(state, action)
                    reward = self.grid.reward(state, action, next_state)
                    if self.grid.is_terminal(next_state):
                        achieve_value = reward
                    else: 
                        achieve_value = reward + self.gamma * self.value_matrix[next_state[0]][next_state[1]]
                    if achieve_value > best_val:
                        best, best_val = action, achieve_value
                else:
                    ## TO-DO
                    print("ERROR!")
        return best

    # For ValueAgent, calling Update on a sample will just call update. 
    def update_sample(self, state, action, next_state, reward):
        self.update(state)

    def update(self, state):
        if self.grid.is_terminal(state):
            return
        actions = self.grid.get_actions(state)
        best_val = np.NINF
        for action, _next in enumerate(actions):
            if self.grid.simple:
                next_state = self.grid.transition(state, action)
                reward = self.grid.reward(state, action, next_state)
                if self.grid.is_terminal(next_state):
                    achieve_value = reward
                else: 
                    achieve_value = reward + self.gamma * self.value_matrix[next_state[0]][next_state[1]]
                best_val = max(best_val, achieve_value)
            else:
                ## TO-DO
                print("ERROR!")
        self.value_matrix[state[0]][state[1]] = best_val
        return best_val
  
    # Perform a value iteration pass.
    def update_all(self):
        for x in range(self.x):
            for y in range(self.y):
                self.update((x, y))
    
    # Run value iteration til convergence. Return the number of full passes executed.
    def run_value_iteration(self):
        counts = 0
        threshold = 10 ** -5
        delta = 1
        while delta > threshold:
            counts += 1
            delta = 0
            # Iterate over every state.
            for x in range(self.x):
                for y in range(self.y):
                    state_value = self.value_matrix[x][y]
                    new_value = self.update((x, y))
                    # Record the biggest change.
                    delta = max(delta, np.abs(state_value - new_value))
        # Once the largest change in value for a pass is less than threshold, assume convergence.
        return counts
        