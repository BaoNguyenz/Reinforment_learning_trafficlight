import numpy as np
from environment import TrafficEnv
from q_learning_agent import QLearningAgent
from config import EPSILON_MIN

def discretize_state(state, bins):
    discretized = []
    for i, val in enumerate(state):
        digitized_val = np.digitize(val, bins)
        discretized.append(digitized_val)
    return tuple(discretized)

if __name__ == "__main__":
    queue_bins = [0, 5, 10, 15]
    num_queue_bins = len(queue_bins) + 1
    state_space_size = tuple([num_queue_bins] * 12)
    
    model_load_path = r'models/q_table_traffic_model.npy'
    
    env = TrafficEnv(use_gui=True)  
    agent = QLearningAgent(state_space_size)
    agent.load(model_load_path)
    agent.epsilon = EPSILON_MIN  

    state, _ = env.reset()
    discretized_state = discretize_state(state, queue_bins)
    total_reward = 0
    done = False
    step = 0

    while not done:
        action = np.argmax(agent.q_table[discretized_state])  
        next_state, reward, done, _, _ = env.step(action)
        next_discretized_state = discretize_state(next_state, queue_bins)
        discretized_state = next_discretized_state
        total_reward += reward
        step += 1

    print(f"Evaluate Q-learning: Total reward: {total_reward:.2f} in {step} steps")
    env.close()
