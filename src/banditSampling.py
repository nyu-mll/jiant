import random
import numpy as np
import matplotlib.pyplot as plt

class Bandit():
    def __init__(self, actions,
                 stepSize = 0.3, initialQ=0.0,
                 explore_method=None, #'epsilon','gradient'
                 temp=1.0,
                 epsilon=0.2):

        self.mapping = dict(enumerate(actions))
        self.k = len(actions)
        #print (self.mapping)
        self.indices = [i for i in range(self.k)]

        # update: constant step size
        self.stepSize = stepSize
        # intialize action value: optimistic if large value
        self.Q = np.array([float(initialQ)] * self.k)

        self.explore_method = explore_method
        # boltzmann exploration: softmax temp
        if explore_method == 'gradient':
            self.temp = temp
        # epsilon greedy exploration: epsilon
        if explore_method == 'epsilon':
            self.epsilon = epsilon

        # keep trace
        self.action = None
        self.reward = None

    def chooseAction(self):
        if self.explore_method == 'gradient':
            self.chooseAction_gradient()
        if self.explore_method == 'epsilon':
            self.chooseAction_epsilonGreedy()

    def chooseAction_gradient(self):
        # softmax(nparray,temp)
        nparray_exp = np.exp(self.Q/self.temp)
        action_prob = nparray_exp/sum(nparray_exp)
        self.action = random.choices(self.indices,action_prob,k=1)[0]

    def chooseAction_epsilonGreedy(self):
        # optimal_index = np.argmax(self.Q)
        # optimal action: random tie breaking for equal val
        optimal_index = np.random.choice(np.where(self.Q == self.Q.max())[0])
        action_prob = [self.epsilon/(k-1)] * self.k
        action_prob [optimal_index] = 1-self.epsilon
        self.action = random.choices(self.indices,action_prob,k=1)[0]

    def update_actionValue(self, reward):
        self.reward = reward
        index = self.action
        self.Q[index] += self.stepSize * (reward - self.Q[index])
