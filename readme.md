# GridWorld DQN 实现

## 简介

本项目实现了一个基于 DQN（深度 Q 网络）算法的简单强化学习示例，名为 **GridWorld**。在这个 5x5 的网格世界中，智能体需要从起点移动到终点，途中需要避开障碍物。通过训练，智能体将学习如何以最优路径到达目标位置。DQN算法的实现参考了论文《Playing Atari with Deep Reinforcement Learning》，链接为https://arxiv.org/abs/1312.5602

## 算法原理

### 强化学习

强化学习是一种机器学习方法，智能体通过与环境的交互来学习策略，以最大化累积奖励。在每个时间步，智能体执行一个动作，环境返回新的状态和奖励。

### Q-Learning

Q-Learning 是一种基于值函数的强化学习算法，用于估计每个状态-动作对的价值（Q 值）。其更新公式为：

Q(s, a) = Q(s, a) + α * [r + γ * max Q(s', a') - Q(s, a)]


其中：

- `Q(s, a)`：状态 `s` 下执行动作 `a` 的价值估计，由Q network给出。
- `max Q(s', a')`：在新状态 `s'` 下所有可能动作的最大 Q 值，即下一步的最优预期价值，由目标网络给出。
- `α`：学习率，控制更新的步长。
- `r`：即时奖励。
- `γ`：折扣因子，平衡即时奖励和未来奖励的影响。
- `s'`：执行动作 `a` 后到达的新状态。
- `a'`：在新状态 `s'` 下可能的动作。

### 深度 Q 网络（DQN）

DQN 使用神经网络来近似 Q 值函数，能够处理高维、连续的状态空间。主要特点包括：

- **神经网络近似**：使用神经网络来估计 Q 值。
- **经验回放**：将经验（状态、动作、奖励、下一状态）存储在记忆库中，训练时随机采样，打破数据相关性。
- **目标网络**：引入一个目标网络来计算目标 Q 值，定期更新，增加训练的稳定性。

## 实现细节

### 环境设计

- **网格大小**：5x5。
- **起点**：坐标 (0, 0)。
- **终点**：坐标 (4, 4)。
- **障碍物**：位于 (1,1)，(2,2)，(3,3)。
- **动作**：上（0），下（1），左（2），右（3）。
- **奖励（奖励设计非常重要，此处可以尝试把移动一步的奖励设为0进行对比）**：
  - 每移动一步：-1。
  - 到达终点：+10。
  - 撞到障碍物：-10，并结束回合。

### 智能体设计

- **神经网络结构**：
  - 输入层：2 个节点（智能体的位置坐标）。
  - 隐藏层：两个全连接层，每层 24 个神经元，激活函数为 ReLU。
  - 输出层：4 个节点（对应 4 个动作的 Q 值）。

- **策略**：ε-贪心策略（ε-greedy），以概率 ε 选择随机动作，以概率 1 - ε 选择当前最优动作。ε 随着训练逐渐减小。

- **超参数**：
  - 学习率：0.0005
  - 折扣因子 γ：0.95
  - 探索率 ε：初始值 1.0，最小值 0.01，衰减率 0.995
  - 记忆库大小：5000
  - 批次大小：128
 
- **损失函数设计**：使用平方误差损失，Loss=（r + γ * max Q(s', a') - Q(s, a)）^2。这种损失函数的设计隐含了“使得Q网络估计的Q(s,a)向r + γ * max Q(s', a')逼近”这一直观感受，与Q-learning的目标相同。

### 训练过程

1. **初始化环境和智能体**。
2. **重复以下步骤直至达到设定的回合数**：
   - 重置环境，获取初始状态。
   - 在每个时间步：
     - 根据当前状态选择动作。
     - 执行动作，获得下一状态、奖励和是否结束。
     - 将经验存储到记忆库。
     - 当记忆库中有足够的经验时，开始训练：
       - 从记忆库中随机采样一批经验。
       - 计算目标 Q 值：
         - 如果是终止状态，目标 Q 值等于即时奖励 `r`。
         - 如果不是终止状态，目标 Q 值等于 `r + γ * max Q_target(s', a')`。
       - 计算预测 Q 值和目标 Q 值之间的损失（均方误差）。
       - 反向传播，更新神经网络参数。
     - 更新当前状态。
     - 如果达到结束状态，跳出循环。
   - 减小探索率 ε。
   - 定期更新目标网络，将主网络的参数复制到目标网络。

### 代码结构

- `agent.py`： 智能体类，包括神经网络结构、策略和训练过程。
- `environment.py`：环境类，包括网格世界的设计和状态转移。
- `main.py`：主程序，用于初始化环境和智能体，以及训练过程。

### 额外说明
在训练过程中，Q-network的训练有可能会出现波动幅度大的情况，这是由于目标网络更新幅度大导致策略变化大导致的，为了缓解
这个问题，我建议可以适当减小目标网络的更新频率，比如每10次更新一次目标网络。以及使用更小的学习率，比如0.0001。或者对于
target network的更新可以使用soft update的方式，即每次更新时只更新一部分参数，比如只更新0.1的参数（此处没有实现，您可以自主实现）。
