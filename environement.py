import matplotlib.pyplot as plt
import numpy as np
import random

# Environment setup
class FrozenLakeEnv:
    def __init__(self, size=5):
        self.size = size
        self.reset()
    
    def reset(self):
        self.grid = np.full((self.size, self.size), '.')
        
        # Place Start and Goal
        self.start = (0, 0)
        self.goal = (self.size-1, self.size-1)
        self.grid[self.start] = 'S'
        self.grid[self.goal] = 'G'
        
        # Place some random holes
        num_holes = (self.size**2) // 5  # ~20% holes
        for _ in range(num_holes):
            x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
            if (x, y) not in [self.start, self.goal]:
                self.grid[x, y] = 'H'
        
        # Agent at start
        self.agent_pos = self.start
        return self.agent_pos
    
    def step(self, action):
        # Actions: 0=up, 1=down, 2=left, 3=right
        x, y = self.agent_pos
        if action == 0 and x > 0: x -= 1
        elif action == 1 and x < self.size-1: x += 1
        elif action == 2 and y > 0: y -= 1
        elif action == 3 and y < self.size-1: y += 1
        
        self.agent_pos = (x, y)
        reward, done = 0, False
        
        if self.agent_pos == self.goal:
            reward, done = 1, True
        elif self.grid[self.agent_pos] == 'H':
            reward, done = -1, True
        
        return self.agent_pos, reward, done
    
    def render(self, path=[]):
        grid_vis = self.grid.copy()
        for (x, y) in path:
            if grid_vis[x, y] == '.':
                grid_vis[x, y] = '*'
        grid_vis[self.agent_pos] = 'A'
        
        plt.imshow([[{'S':0,'G':1,'H':2,'.':3,'*':4,'A':5}[c] for c in row] for row in grid_vis],
                   cmap=plt.cm.get_cmap('Set1', 6))
        plt.xticks([]); plt.yticks([])
        plt.show()

# Example usage
env = FrozenLakeEnv(size=6)
env.reset()

done = False
path = []
while not done:
    action = random.choice([0,1,2,3])  # random agent for now
    state, reward, done = env.step(action)
    path.append(state)
    env.render(path)
