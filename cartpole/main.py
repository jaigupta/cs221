import gym
import math
import random
import numpy as np
from time import sleep
import tensorflow as tf
import inspect
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


env = gym.make('CartPole-v0')

def CreateKerasModel(total_features):
    model = Sequential()
    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='linear'))
    return model

class QLearning(object):
    def __init__(self, num_features, actions):
        total_features=num_features*len(actions) + 1
        self.weights = np.zeros(total_features)
        self.actions = actions
        self.discount = 0.7
        self.gamma = 0.4
        self.epsilon = 0.2
        self.model = CreateKerasModel(total_features)
        self.optimizer = tf.optimizers.Adam()
        self.batch_size = 100
        self.x_train = []
        self.y_train = []

    def getAction(self, state, step=0):
        if random.random() < self.getEpsilon(step):
            return self.actions[int(random.random()*2)]
        scores = []
        for a in self.actions:
            scores.append((self.getQ(step, state, a), a))
        return max(scores)[1]

    def createFeatures(self, step, state, action):
        features=[step]
        for ai in self.actions:
            if ai == action:
                features = np.append(features, state)
            else:
                features = np.append(features, np.zeros(len(state)))
        return features


    def getQ(self, step, state, action):
        features = self.createFeatures(step, state, action)
        return self.model(tf.stack([features, features]), training=False).numpy()[0][0]

        return np.dot(self.weights, features)

    def getGamma(self, step):
        return  self.gamma

    def getEpsilon(self, step):
        return self.epsilon/(1+math.sqrt(step/10.0))

    def takeFeedback(self, step, state, action, reward, newState):
        vNewState = max(self.getQ(step, newState, ai) for ai in self.actions)
        err = self.getQ(step, state, action) - reward - self.discount*vNewState
        self.weights = self.weights - self.gamma * err * self.createFeatures(step, state, action)

        math.abs = abs
        cart_pos_reward = math.abs(state[0]) - math.abs(newState[0])
        cart_velocity_reward = math.abs(state[1]) - math.abs(newState[1])
        pole_angle_reward = math.abs(state[2]) - math.abs(newState[2])
        pole_velocity_reward = math.abs(state[3]) - math.abs(newState[3])
        reward = (0.05*cart_pos_reward+0.005*cart_velocity_reward+pole_angle_reward+0.005*pole_velocity_reward)

        self.x_train.append(self.createFeatures(step, state, action))
        self.y_train.append(reward+self.discount*max(vNewState, 0))
        if len(self.x_train) >= self.batch_size:
            # print("Evaluating", len(self.x_train))
            self.x_train = tf.stack(self.x_train)
            self.y_train = tf.stack(self.y_train)
            with tf.GradientTape() as tape:
                pred = self.model(self.x_train, training=True)
                loss = tf.losses.mean_squared_error(pred, self.y_train)
                print("Sum pred:", sum(pred.numpy()), "actual: ", sum(self.y_train.numpy()))
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            self.x_train = []
            self.y_train = []
        

globalReward = 0
numIters = 4000

rl = QLearning(len(env.reset()), [0, 1])
for i in range(numIters):
    state = env.reset()
    done = False
    totalReward = 0
    while not done:
        if i % 50 < 3:
            env.render()
            sleep(0.03)
        action = rl.getAction(state, i)
        newState, reward, done, info = env.step(action)
        rl.takeFeedback(i, state, action, reward, newState)
        state = newState
        totalReward += reward
    print("Reward: ", totalReward)
    globalReward += totalReward


env.close()
print('Reward avg:', globalReward/float(numIters))
