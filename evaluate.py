import numpy as np
from environment import TrafficEnv
from agent import DQNAgent
from config import MODEL_SAVE_PATH
import time

if __name__ == "__main__":
    env = TrafficEnv(use_gui=True)
    agent = DQNAgent()
    
    try:
        agent.load(MODEL_SAVE_PATH)
        agent.epsilon = 0.0
        print("Load model successfully.")
    except Exception as e:
        print(f"Can't load model: {e}")
        env.close()
        exit()

    state, _ = env.reset()
    state = np.reshape(state, [1, agent.state_size])
    total_reward = 0
    done = False
    
    while not done:
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        next_state = np.reshape(next_state, [1, agent.state_size])
        state = next_state
        total_reward += reward
        time.sleep(0.05)

    print(f"Evaluate completed. Total reward: {total_reward:.2f}")
    env.close()