import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_training_results(history_path=r'E:\LET ME COOK\REL301m\dqn_traffic_project\models\dqn_traffic_model_3lane.h5', save_fig_path='training_results.png'):
    if not os.path.exists(history_path):
        print(f"Error, cannot find file '{history_path}'.")
        return

    history = np.load(history_path, allow_pickle=True).item()
    rewards = history['rewards']
    epsilons = history.get('epsilons', [])
    losses = history.get('losses', [])
    vehicles_remaining = history.get('vehicles_remaining', [])
    episodes = range(1, len(rewards) + 1)

    reward_series = pd.Series(rewards)
    moving_avg_rewards = reward_series.rolling(window=10).mean()
    loss_series = pd.Series(losses)
    moving_avg_losses = loss_series.rolling(window=10).mean()
    vehicles_series = pd.Series(vehicles_remaining)
    moving_avg_vehicles = vehicles_series.rolling(window=10).mean() if len(vehicles_remaining) else None

    fig_rows = 4 if len(vehicles_remaining) else 3
    fig, axs = plt.subplots(fig_rows, 1, figsize=(12, 5*fig_rows), sharex=False)
    fig.suptitle('DQN Traffic Training Results', fontsize=16)

    axs[0].plot(episodes, rewards, label='Reward per Episode', alpha=0.5)
    axs[0].plot(episodes, moving_avg_rewards, label='Moving Average (10 Episodes)', color='red', linewidth=2)
    axs[0].set_ylabel('Total Reward')
    axs[0].set_title('Reward Over Time')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(episodes, epsilons, label='Epsilon Value', color='green')
    axs[1].set_ylabel('Epsilon')
    axs[1].set_title('Epsilon Decay (Exploration Strategy)')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(losses, label='Loss', color='purple', alpha=0.4)
    axs[2].plot(loss_series.index, moving_avg_losses, label='Moving Avg Loss', color='orange', linewidth=2)
    axs[2].set_xlabel('Training Step')
    axs[2].set_ylabel('Loss')
    axs[2].set_title('Model Convergence (Loss Function)')
    axs[2].legend()
    axs[2].grid(True)

    if len(vehicles_remaining):
        axs[3].plot(episodes, vehicles_remaining, label='Vehicles Remaining', color='blue', alpha=0.7)
        axs[3].plot(episodes, moving_avg_vehicles, label='Moving Avg (10)', color='red', linewidth=2)
        axs[3].set_ylabel('Vehicles Remaining')
        axs[3].set_title('Vehicles Remaining per Episode')
        axs[3].legend()
        axs[3].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_fig_path)
    print(f"Chart saved to '{save_fig_path}'")
    plt.show()

if __name__ == '__main__':
    plot_training_results()
