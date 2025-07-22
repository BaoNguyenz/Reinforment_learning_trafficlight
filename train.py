import os
import numpy as np
from environment import TrafficEnv
from agent import DQNAgent
from config import EPISODES, BATCH_SIZE, USE_GUI, MODEL_SAVE_PATH, EARLY_STOP_PATIENCE
from collections import deque
import traci

save_dir = os.path.dirname(MODEL_SAVE_PATH)
if save_dir and not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"auto make directorry: {save_dir}")

if __name__ == "__main__":
    env = TrafficEnv(use_gui=USE_GUI)
    agent = DQNAgent()
    reward_history = []
    epsilon_history = []
    loss_history = deque(maxlen=10000)
    vehicles_remaining_history = []

    best_avg_reward = -float("inf")
    no_improve_count = 0

    for e in range(EPISODES):
        state, _ = env.reset()
        state = np.reshape(state, [1, agent.state_size])
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.reshape(next_state, [1, agent.state_size])
            total_reward += reward
            agent.remember(state[0], action, reward, next_state[0], done)
            state = next_state
            loss = agent.replay()
            if loss > 0:
                loss_history.append(loss)
        
        reward_history.append(total_reward)
        epsilon_history.append(agent.epsilon)

        try:
            vehicle_remaining = traci.vehicle.getIDCount()
        except:
            vehicle_remaining = 0
        vehicles_remaining_history.append(vehicle_remaining)
        
        print(f"Episode: {e+1}/{EPISODES}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
        
        if (e + 1) % 10 == 0:
            agent.save(MODEL_SAVE_PATH)
            print(f"Model has been saved at episode {e+1}")

        if e >= 10:
            avg_reward = np.mean(reward_history[-10:])
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= EARLY_STOP_PATIENCE:
                    print("Early stopping: no improvement in average reward.")
                    break
                    
    history_data = {
        'rewards': reward_history,
        'epsilons': epsilon_history,
        'losses': list(loss_history),
        'vehicles_remaining': vehicles_remaining_history
    }
    np.save(r'models\training_history_3lane.npy', history_data)
    print("Saved training history.")
    env.close()
    print("Training completed.")