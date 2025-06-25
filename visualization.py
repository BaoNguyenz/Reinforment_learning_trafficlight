import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_training_results(history_path='models/training_history_3lane.npy', save_fig_path='training_results.png'):
    if not os.path.exists(history_path):
        print(f"Error, cannot find file '{history_path}'.")
        return

    history = np.load(history_path, allow_pickle=True).item()
    rewards = history['rewards']
    epsilons = history.get('epsilons', [])
    losses = history.get('losses', [])
    episodes = range(1, len(rewards) + 1)

    reward_series = pd.Series(rewards)
    moving_avg_rewards = reward_series.rolling(window=10).mean()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    fig.suptitle('DQN Traffic Training Results', fontsize=16)

    ax1.plot(episodes, rewards, label='Reward per Episode', alpha=0.5)
    ax1.plot(episodes, moving_avg_rewards, label='Moving Average (10 Episodes)', color='red', linewidth=2)
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Reward Over Time')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(episodes, epsilons, label='Epsilon Value', color='green')
    ax2.set_ylabel('Epsilon')
    ax2.set_title('Epsilon Decay (Exploration Strategy)')
    ax2.legend()
    ax2.grid(True)

    ax3.plot(losses, label='Loss', color='purple', alpha=0.7)
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Loss')
    ax3.set_title('Model Convergence (Loss Function)')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_fig_path)
    print(f"Chart saved to '{save_fig_path}'")
    plt.show()

if __name__ == '__main__':
    plot_training_results()
