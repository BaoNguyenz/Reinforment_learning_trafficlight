import numpy as np
from environment import TrafficEnv
from agent import DQNAgent
import time
import traci

if __name__ == "__main__":
    env = TrafficEnv(use_gui=True)
    agent = DQNAgent()
    
    agent.epsilon = 0.0
    print

    state, _ = env.reset()
    state = np.reshape(state, [1, agent.state_size])
    total_reward = 0
    done = False
    step = 0
    
    while not done:
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        next_state = np.reshape(next_state, [1, agent.state_size])
        state = next_state
        total_reward += reward
        time.sleep(0.05)

    print(f"Evaluate before training, Total reward: {total_reward:.2f}")
    env.close()