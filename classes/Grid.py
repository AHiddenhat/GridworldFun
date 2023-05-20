import numpy as np

DATA = ["start", "terminal", "wall"]
# Start is where the agent begins. 
# Terminals are states that only lead to the terminal state. (state, reward)
# Walls are inaccessible locations to the state space. (state)
# Random specifies the chance that random movement will occur instead of the chosen movement.
    # p, where p in [0, 1].
# We define the "terminal state" as the tuple (-1, -1), which has value 0, only a looping action, and no more living reward.

class Grid():
    # The Grid class assumes a 2D representation of the State space, 
    # where the only actions to take are up, down, left, right. 
    def __init__(self, dimensions: tuple, data: dict, random=0, living_reward=0): 
        self.x = dimensions[0]
        self.y = dimensions[1]
        self.dimensions = dimensions
        self.state_matrix = np.zeros(self.dimensions)
        self.random = random
        if random != 0:
            self.build_expectation_matrix()
        self.living_reward = living_reward
        self.data = data
        self.construct(data)

    # The Construct method autocreates the 
    # state matrix, transition function, and reward function, 
    # given paramater data, a dictionary. 
    def construct(self, data):
        self.start = data["start"]
        for state, reward in data["terminal"]:
            self.state_matrix[state[0]][state[1]] = reward
        for state in data["wall"]:
            self.state_matrix[state[0]][state[1]] = "W"
    
    # Get next state. Action is an integer that specifies the action taken.
    def transition(self, state, action, deterministic=False):
        actions = self.get_actions(state)
        if np.random.random() > self.random or deterministic: # Check for random transition.
            return actions[action]

        probability = np.random.random()
        ind, thresh = 0, 1.0 / len(actions) 
        while probability > thresh:
            ind += 1
            thresh += 1.0 / len(actions)
        return actions[ind]
    
    # Recall action is an index into the list implied by self.get_actions(state)
    def reward(self, state: tuple, action: int, next_state: tuple):
        if self.is_terminal(state):
            return 0
        return self.state_matrix[state[0]][state[1]] + self.living_reward

    def expected_reward(self, state: tuple, action: int):
        return self.expectation_matrix[state[0], state[1], action]
    
    def build_expectation_matrix(self):
        try:
            d = self.d
        except AttributeError:
            d = 4

        self.expectation_matrix = np.zeros((self.x, self.y, d))
        for x in range(self.x):
            for y in range(self.y):
                for a in range(len(self.get_actions((x, y)))):
                    self.expectation_matrix[x][y][a] = self.build_expectation((x, y), a)
    
    def build_expectation(self, state: tuple, action: int):
        if self.random == 0:
            return self.reward(state, action, self.transition(state, action))

        expect_reward = (1.0 - self.random) * self.reward(state, action, self.transition(state, action, True))
        actions = self.get_actions(state)
        for a in actions:
            expect_reward += (self.random / len(actions)) * self.reward(state, a, self.transition(state, a, True))
        return expect_reward
            
    def is_terminal(self, state):
        return state == (-1, -1)
    
    def is_preterminal(self, state):
        return state in [x for x, _ in self.data["terminal"]]
        
    # Return a list of actions and their next states; assuming a deterministic model.
    def get_actions(self, state):
        if self.is_terminal(state) or self.is_preterminal(state):
            return [(-1, -1)]

        answer = []
        x = state[0]
        y = state[1]
        if y > 0 and self.state_matrix[x][y - 1] != "W":
            answer.append((x, y - 1))
        if x < self.x - 1 and self.state_matrix[x + 1][y] != "W":
            answer.append((x + 1, y))
        if y < self.y - 1 and self.state_matrix[x][y + 1] != "W":
            answer.append((x, y + 1))
        if x > 0 and self.state_matrix[x - 1][y] != "W":
            answer.append((x - 1, y))
        return answer



      