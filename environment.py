import numpy as np

class GridWorld:
    def __init__(self):
        self.size = 5
        self.reset()
        self.actions = [0, 1, 2, 3]  # 上，下，左，右

    def reset(self):
        self.grid = np.zeros((self.size, self.size))
        self.position = [0, 0]
        self.goal = [4, 4]
        self.obstacles = [[1,1], [2,2], [3,3]]
        for obs in self.obstacles:
            self.grid[obs[0], obs[1]] = -1
        self.grid[self.goal[0], self.goal[1]] = 1
        return self.position

    def step(self, action):
        x, y = self.position
        if action == 0 and x > 0:
            x -= 1  # 上
        elif action == 1 and x < self.size -1:
            x += 1  # 下
        elif action == 2 and y > 0:
            y -= 1  # 左
        elif action == 3 and y < self.size -1:
            y += 1  # 右

        self.position = [x, y]

        reward = -1
        done = False

        if self.position == self.goal:
            reward = 10
            done = True
        elif self.position in self.obstacles:
            reward = -10
            done = True

        return self.position, reward, done

    def render(self):
        grid = np.zeros((self.size, self.size), dtype=str)
        grid[:] = '.'
        for obs in self.obstacles:
            grid[obs[0], obs[1]] = 'X'
        grid[self.goal[0], self.goal[1]] = 'G'
        grid[self.position[0], self.position[1]] = 'A'
        print('\n'.join(' '.join(row) for row in grid))
        print()
