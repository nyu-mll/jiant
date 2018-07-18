import random
import numpy as np
from allennlp.common import Params
import copy


class Bandit():
    def __init__(self, actions,
                 stepSize = 0.3, initialQ=0.0,
                 explore_method=None, # 'epsilon', 'boltzmann'
                 temp=1.0,
                 epsilon=0.2,
                 val_batch =1, val_generation = 'static'): # 'static', 'dynamic'

        self.mapping = dict(enumerate(actions))
        self.k = len(actions)
        self.indices = [i for i in range(self.k)]

        # constant step-size for update
        self.stepSize = stepSize
        # intialize action value: optimistic if large value
        self.Q = np.array([float(initialQ)] * self.k)

        # method specific param
        self.explore_method = explore_method
        self.temp = None
        self.epsilon = None
        if explore_method == 'boltzmann':
            self.temp = temp
        elif explore_method == 'epsilon':
            self.epsilon = epsilon

        # keep trace
        self.action = None
        self.reward = None

        # validation control
        self.val_batch = val_batch
        self.val_generation = val_generation

    def chooseAction(self):
        if self.explore_method == 'boltzmann':
            self.chooseAction_boltzmann()
        if self.explore_method == 'epsilon':
            self.chooseAction_epsilonGreedy()

    def chooseAction_boltzmann(self):
        action_prob = softmax(self.temp,self.Q)
        self.action = random.choices(self.indices,action_prob,k=1)[0]

    def chooseAction_epsilonGreedy(self):
        # optimal action: random tie breaking for equal value
        optimal_index = np.random.choice(np.where(self.Q == self.Q.max())[0])
        action_prob = [self.epsilon/(self.k-1)] * self.k
        action_prob [optimal_index] = 1 - self.epsilon
        self.action = random.choices(self.indices,action_prob,k=1)[0]

    def update_actionValue(self, reward):
        self.reward = reward
        index = self.action
        # exponential recency-weighted average
        self.Q[index] += self.stepSize * (reward - self.Q[index])

    @classmethod
    def from_params(cls,actions,params):
        ''' Generate trainer from parameters.  '''

        stepSize = params.pop("stepSize", 0.3)
        initialQ = params.pop("initialQ", 0)
        explore_method = params.pop("explore_method", "boltzmann")
        temp = params.pop("temp", 1)
        epsilon = params.pop("epsilon", 0.2)
        val_batch = params.pop("val_batch",1)
        val_generation = params.pop("val_generation","static")

        params.assert_empty(cls.__name__)
        return Bandit(actions,
                     stepSize = stepSize, initialQ=initialQ,
                     explore_method=explore_method,
                     temp=temp,
                     epsilon=epsilon,
                     val_batch = val_batch,
                     val_generation= val_generation)


def build_bandit(params, actions):
    ''' Build a bandit '''
    bandit_params = Params({'stepSize': params['stepSize'],
                           'initialQ': params['initialQ'],
                           'explore_method': params['explore_method'],
                           'temp': params['temp'],
                           'epsilon': params['epsilon'],
                           'val_batch':params['val_batch'],
                           'val_generation':params['val_generation']})

    bandit = Bandit.from_params(actions,copy.deepcopy(bandit_params))
    return bandit


def build_bandit_params(args):
    ''' Build bandit parameters '''
    params = {}
    opts = ['stepSize', 'initialQ', 'explore_method','temp', 'epsilon','val_batch','val_generation']
    for attr in opts:
        params[attr] = getattr(args, attr)

    return Params(params)


def softmax(temp, nparray):
    #subtract max for numerical stability
    nparray = nparray - np.max(nparray)
    nparray_exp = np.exp(nparray/temp)
    return (nparray_exp/sum(nparray_exp))
