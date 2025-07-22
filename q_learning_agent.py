import numpy as np
import random
from config import ACTION_SIZE, LEARNING_RATE, GAMMA

class QLearningAgent:
    def __init__(self, state_space_size, action_space_size=ACTION_SIZE):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        
        self.q_table = np.zeros(self.state_space_size + (self.action_space_size,))
        
        self.learning_rate = LEARNING_RATE
        self.gamma = GAMMA
        
        self.epsilon = 1.0

    def act(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randrange(self.action_space_size) # Khám phá
        return np.argmax(self.q_table[state]) # Khai thác

    def update(self, state, action, reward, next_state):
        old_value = self.q_table[state + (action,)]
        next_max = np.max(self.q_table[next_state])
        
        new_value = old_value + self.learning_rate * (reward + self.gamma * next_max - old_value)
        self.q_table[state + (action,)] = new_value

    def save(self, name):
        np.save(name, self.q_table)

    def load(self, name):
        self.q_table = np.load(name)