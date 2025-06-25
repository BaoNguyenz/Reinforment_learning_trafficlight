import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from config import (STATE_SIZE, ACTION_SIZE, MEMORY_SIZE, GAMMA, LEARNING_RATE,
                    EPSILON_INIT, EPSILON_MIN, EPSILON_DECAY, BATCH_SIZE)

class DQNAgent:
    def __init__(self):
        self.state_size = STATE_SIZE
        self.action_size = ACTION_SIZE
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.gamma = GAMMA
        self.epsilon = EPSILON_INIT
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.learning_rate = LEARNING_RATE
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return 0.0
        minibatch = random.sample(self.memory, BATCH_SIZE)
        states = np.array([transition[0] for transition in minibatch]).reshape(BATCH_SIZE, self.state_size)
        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch]).reshape(BATCH_SIZE, self.state_size)
        dones = np.array([transition[4] for transition in minibatch])
        target_q_next = self.target_model.predict(next_states, verbose=0)
        targets = rewards + self.gamma * np.amax(target_q_next, axis=1) * (1 - dones)
        target_q_current = self.model.predict(states, verbose=0)
        target_q_current[np.arange(BATCH_SIZE), actions] = targets
        history = self.model.fit(states, target_q_current, epochs=1, verbose=0)
        loss = history.history['loss'][0]

        del history
        K.clear_session()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

    def load(self, name):
        self.model.load_weights(name)
        self.update_target_model()

    def save(self, name):
        self.model.save_weights(name)