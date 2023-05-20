import numpy as np

class Agent():
    def __init__(self, grid, epsilon=0, gamma=1):
        self.grid = grid
        self.position = grid.data["start"]
        self.dimensions = grid.dimensions
        self.x = self.dimensions[0]
        self.y = self.dimensions[1] 
        self.epsilon = epsilon # Chance to take random action instead of policy action.
        self.gamma = gamma # Discount for future rewards.
  
    def reset(self):
        self.position = self.grid.data["start"]
  
    # When eps=True, follows epsilon-greedy. Otherwise, follows policy.
    def get_action(self, state, eps=True) -> int:
        pass
  
    def update_sample(self, state, action, next_state, reward):
        pass

    # Run an episode. If Update, will update the policy of the agent. If eps, use epsilon greedy.
    def run_episode(self, update=True, eps=True):
        buffer = []
        while not self.grid.is_terminal(self.position):
            state = self.position
            action = self.get_action(self.agent.position, eps)
            next_state, reward = self.grid.transition(self.position, action)
            self.position = next_state
            if update:
                self.update_sample(state, action, next_state, reward)
            buffer.append([state, action, next_state, reward])
        self.reset()
        return buffer

###

class ValueAgent(Agent):
    def __init__(self, grid, random_init=False, epsilon=1, gamma=0.9):
        super().__init__(grid, epsilon, gamma)
        if random_init: 
            self.value_matrix = np.random.rand(grid.dimensions[0], grid.dimensions[1])
        else:
            self.value_matrix = np.zeros(grid.dimensions)

    def get_action(self, state, eps=True) -> int:
        actions = self.grid.get_actions(state)
        if eps and np.random.random() < self.epsilon:
            return np.random.randint(0, len(actions))
        else:
            best, best_val = 0, np.NINF
            for action, _next in enumerate(actions):
                if self.grid.random == 0:
                    next_state = self.grid.transition(state, action)
                    reward = self.grid.reward(state, action, next_state)
                    if self.grid.is_terminal(next_state):
                        achieve_value = reward
                    else: 
                        achieve_value = reward + self.gamma * self.value_matrix[next_state[0]][next_state[1]]
                    if achieve_value > best_val:
                        best, best_val = ind, achieve_value
                if self.grid.random != 0:
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
            if self.grid.random == 0:
                next_state = self.grid.transition(state, action)
                reward = self.grid.reward(state, action, next_state)
                if self.grid.is_terminal(next_state):
                    achieve_value = reward
                else: 
                    achieve_value = reward + self.gamma * self.value_matrix[next_state[0]][next_state[1]]
                best_val = max(best_val, achieve_value)
            if self.grid.random != 0:
                ## TO-DO
                print("ERROR!")
        self.value_matrix[state[0]][state[1]] = best_val
  
    # Perform a value iteration pass.
    def update_all(self):
        for x in range(self.x):
            for y in range(self.y):
                self.update((x, y))


