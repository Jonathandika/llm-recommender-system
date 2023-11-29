import random
import argparse
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import losses
import itertools as it
import pandas as pd
import keras
from keras import backend as K
from collections import Counter
import heapq
from tqdm.notebook import tqdm
import tensorflow as tf

parser = argparse.ArgumentParser()

parser.add_argument('--Model_Type', type=int, default=1, help='0 = Initial Model & 1 = Target Model')
parser.add_argument('--State_Size', type=int, default=2, help='number of articles previously read by a user')
parser.add_argument('--batch', type=int, default=5, help='input batch size')
parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for per user')
parser.add_argument('--LR', type=float, default=0.001, help='learning rate')
parser.add_argument('--deque_length', type=int, default=2000, help='memory to remember')
parser.add_argument('--discount', type=float, default=0.95, help='gamma')
parser.add_argument('--eps', type=float, default=1.0, help='epsilon')
parser.add_argument('--eps_decay', type=float, default=0.995, help='epsilon decay')
parser.add_argument('--eps_min', type=float, default=0.01, help='minimum epsilon')

optim = parser.parse_args(args = [])


class DQNAgent:

    # Function that contains all the hyperparameters
    def __init__(self, state_size, action_size, retrain = False):
        """
        Initialize Hyperparameters.

        Parameters:
        arg1 (int): State Size
        arg2 (int): Action Size

        """
        self.state_size = state_size # Total number of states ( Combinations of all the Books )
        self.action_size = action_size # Number of Recommendations
        self.memory = deque(maxlen=optim.deque_length) # A double Q to store the states, actions and rewards
        self.gamma = optim.discount # Discount Factor
        self.epsilon = optim.eps  # Exploration Factor
        self.epsilon_decay = optim.eps_decay # Decay for Exploration Factor
        self.epsilon_min = optim.eps_min
        self.learning_rate = optim.LR
        
        if retrain == True:
            self.model = self._build_model()
        else:
            self.model = keras.models.load_model('tmp_model')

    # Function to build a model
    def _build_model(self):
        """
        Constructs a Neural Network Model.


        Returns:
        list: Returns the Model.

        """
        model = Sequential()
        model.add(Dense(256, input_dim=self.state_size, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
        return model

     # Fuction to store the initial state, the action ( recommendation in this case ) , reward, next state and the termination condition
    def remember(self, state, action, reward, next_state, done):
        """
        Makes a Double Queue to store Variables.

        Parameters:
        arg1 (list): Current State
        arg1 (int): Action
        arg1 (int): Reward
        arg1 (list): Next State
        arg1 (bool): Done

        """
        self.memory.append((state, action, reward, next_state, done))

    # Function deciding the Recommendation. Either takes a random value or takes the maximum from a list of Q-Values
    def act(self, state):
        """
        Takes action/recommendation based on largest Q-Value.

        Parameters:
        arg1 (list): Current State

        Returns:
        int: Returns the action with maximum Q-Value.

        """
        if np.random.rand() <= self.epsilon: # Random Exploration
            return random.randrange(self.action_size)
        act_values = self.model.predict(state) # Exploitation
        return np.argmax(act_values[0])

    # Implementation of Deep Q-Learning
    # Function where actual training occurs and states are passed in batches for training
    def replay(self, batch_size):
        """
        Trains the Model.

        Parameters:
        arg1 (int): Batch Size

        """
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state,done  in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay # Decrease the value of Exploration Factor

    # Implementation of Double Deep Q-Learning
    # After every 'N' iterations, the Target Model is used for predicting the next state
    def replay2(self,batch_size, target_model):
        """
        Trains the Target Model.

        Parameters:
        arg1 (int): Batch Size

        """
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state,done  in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(target_model.predict(next_state)[0]))
            target_f = target_model.predict(state)
            target_f[0][action] = target
            target_model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay