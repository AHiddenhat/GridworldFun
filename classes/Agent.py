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


