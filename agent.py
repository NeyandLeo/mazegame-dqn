import numpy as np
import random
from collections import deque
# 导入必要的深度学习库
import torch
import torch.nn as nn
import torch.optim as optim

class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95    # 折扣因子
        self.epsilon = 1.0   # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005
        self.model = self._build_model()
        self.targetnet = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.update_target()  # 将主网络的参数复制到目标网络
        # 冻结目标网络的参数
        for param in self.targetnet.parameters():
            param.requires_grad = False

    def _build_model(self):
        # 请在此处补全神经网络模型的代码
        # 提示：使用PyTorch构建一个简单的全连接网络
        model = nn.Sequential(
            nn.Linear(2, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, 4)
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(4)
        state = torch.FloatTensor(state)
        act_values = self.model(state)
        return torch.argmax(act_values).item()

    def act_greedy(self, state):
        state = torch.FloatTensor(state)
        act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state)
                target = (reward + self.gamma *
                          torch.max(self.targetnet(next_state)).item())
            state = torch.FloatTensor(state)
            target_f = self.model(state)
            target_f = target_f.detach().clone()
            target_f[action] = target

            # 请在此处补全训练过程的代码
            # 提示：计算损失并执行反向传播
            self.optimizer.zero_grad()
            output = self.model(state)
            loss = self.criterion(output, target_f)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target(self):
        self.targetnet.load_state_dict(self.model.state_dict())