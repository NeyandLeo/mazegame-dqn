from environment import GridWorld
from agent import DQNAgent
import numpy as np

if __name__ == "__main__":
    env = GridWorld()
    agent = DQNAgent()
    episodes = 100
    update_target_freq = 5  # 每隔10个回合更新一次目标网络

    for e in range(episodes):
        state = env.reset()
        state = np.array(state)
        re = 0
        for time_t in range(100):
            # env.render()
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            re+=reward
            next_state = np.array(next_state)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"episode: {e}/{episodes}, time: {time_t}, e: {agent.epsilon:.2}, reward: {re}")
                break
            if len(agent.memory) > 128:
                agent.replay(128)
        if e % update_target_freq == 0:
            agent.update_target()
            print("target_net updated")

    # 测试模型
    state = env.reset()
    state = np.array(state)
    for _ in range(100):
        env.render()
        action = agent.act_greedy(state) #using greedy strategy instead of epsilon greedy strategy when testing
        next_state, reward, done = env.step(action)
        next_state = np.array(next_state)
        state = next_state
        if done:
            break