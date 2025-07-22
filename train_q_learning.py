import os
import numpy as np
from environment import TrafficEnv
from q_learning_agent import QLearningAgent
from config import EPISODES, EPSILON_INIT, EPSILON_MIN, EPSILON_DECAY

def discretize_state(state, bins):

    discretized = []
    for i, val in enumerate(state):
        digitized_val = np.digitize(val, bins)
        discretized.append(digitized_val)
    return tuple(discretized)

if __name__ == "__main__":
    queue_bins = [0, 5, 10, 15] # 0-4 xe, 5-9 xe, 10-14 xe, >= 15 xe
    num_queue_bins = len(queue_bins) + 1
    
    state_space_size = tuple([num_queue_bins] * 12)
    
    env = TrafficEnv(use_gui=False)
    agent = QLearningAgent(state_space_size)
    agent.epsilon = EPSILON_INIT 
    
    reward_history = []
    epsilon_history = []
    
    model_save_path = r'models/q_table_traffic_model.npy'
    save_dir = os.path.dirname(model_save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for e in range(EPISODES):
        state, _ = env.reset()
        discretized_state = discretize_state(state, queue_bins)
        
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(discretized_state)
            next_state, reward, done, _, _ = env.step(action)
            next_discretized_state = discretize_state(next_state, queue_bins)
            
            agent.update(discretized_state, action, reward, next_discretized_state)
            
            discretized_state = next_discretized_state
            total_reward += reward

        if agent.epsilon > EPSILON_MIN:
            agent.epsilon *= EPSILON_DECAY

        reward_history.append(total_reward)
        epsilon_history.append(agent.epsilon)
        print(f"Episode: {e+1}/{EPISODES}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

        if (e + 1) % 10 == 0:
            agent.save(model_save_path)
            print(f"Q-table đã được lưu tại episode {e+1}")
            
    history_data = {
        'rewards': reward_history,
        'epsilons': epsilon_history,
        'losses': [] 
    }
    np.save('models/q_learning_history.npy', history_data)
    print("Đã lưu lại lịch sử huấn luyện Q-Learning.")
    env.close()
    print("Huấn luyện Q-Learning hoàn tất.")