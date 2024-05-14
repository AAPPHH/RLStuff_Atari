import gym
import gym_tetris
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, PReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
from collections import deque
from nes_py.wrappers import JoypadSpace
import gym_tetris.actions
import time
 
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0  
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=self.state_size))
        model.add(Flatten())
        model.add(Dense(24))
        model.add(PReLU())
        model.add(Dense(24))
        model.add(PReLU())
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    env = gym_tetris.make('TetrisA-v3')
    env = JoypadSpace(env, gym_tetris.actions.MOVEMENT)
    state_size = (240, 256, 3)
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    done = False
    batch_size = 32

    render_delay = 5
    last_render_time = time.time()
    render_interval = 100000

    for e in range(1000):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        state = np.reshape(state, [1] + list(state_size))
        for time_step in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            next_state = np.reshape(next_state, [1] + list(state_size))
            reward = reward if not done else -10
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            current_time = time.time()
            if e % render_interval == 0 and current_time - last_render_time > render_delay:
                env.render()
                last_render_time = current_time

            if done:
                print("episode: {}/{}, score: {}, e: {:.2}".format(e, 1000, time_step, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

    env.close()
