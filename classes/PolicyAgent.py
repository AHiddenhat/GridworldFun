import numpy as np
from classes.Agent import Agent

class PolicyAgent(Agent):
    def __init__(self, grid, random_init=False, epsilon=1, gamma=0.9):
        super().__init__(grid, epsilon, gamma)
        if random_init:
            self.value_matrix = np.random.rand(self.x, self.y)
        else:
            self.value_matrix = np.zeros(self.dimensions)
        self.policy_matrix = np.zeros(self.dimensions).astype(int)

    def get_action(self, state, eps=True) -> int:
        actions = self.grid.get_actions(state)
        if eps and np.random.random() < self.epsilon:
            return np.random.randint(0, len(actions))
        
        return self.policy_matrix[state[0]][state[1]]

    def update_sample(self, state, action, next_state, reward):
        self.update()

    def update(self, state):
        if self.grid.is_terminal(state):
            return
        x = state[0]
        y = state[1]
        if self.grid.simple:
            policy_action = self.policy_matrix[x][y]
            next_state = self.grid.transition(state, policy_action)
            reward = self.grid.reward(state, policy_action, next_state)
            if self.grid.is_terminal(next_state):
                self.value_matrix[x][y] = reward
            else:
                self.value_matrix[x][y] = reward + self.gamma * self.value_matrix[next_state[0]][next_state[1]]
        else:
            ## to-do
            print("ERROR")
  
    # Perform a value iteration pass.
    def update_all(self):
        for x in range(self.x):
            for y in range(self.y):
                self.update((x, y))
    
    # Run a full pass of policy improvement, and return whether or not the policy changed.
    def policy_improvement(self):
        improvement = False
        for x in range(self.x):
            for y in range(self.y):
                old_action = self.policy_matrix[x][y]
                state = (x, y)
                best, best_val = -1, np.NINF
                if self.grid.simple:
                    for action, next_state in enumerate(self.grid.get_actions(state)):
                        reward = self.grid.reward(state, action, next_state)
                        if self.grid.is_terminal(next_state):
                            achieve_value = reward
                        else: 
                            achieve_value = reward + self.gamma * self.value_matrix[next_state[0]][next_state[1]]
                        if achieve_value > best_val:
                            best, best_val = action, achieve_value
                self.policy_matrix[x][y] = best
                if old_action != best:
                    improvement = True
        return improvement
    
    # Run policy iteration until convergence. 
    def run_policy_iteration(self):
        counts = 0
        improves = True
        while improves:
            counts += 1
            self.update_all()
            improves = self.policy_improvement()
        return counts