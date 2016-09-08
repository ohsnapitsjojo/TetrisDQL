import sys
sys.path.append("..\preprocessing")
sys.path.append("..\inputSimulator")

import time

import numpy as np
import theano
import theano.tensor as T

import lasagne

from network import build_network
from preprocessor import Preprocessor
from inputSimulator import InputSimulator
from agent import Agent

class PGLearner:
    
    def __init__(self, gamma, learning_rate, rho, epsilon):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.rho = rho
        self.eps = epsilon
        
        self.iS = InputSimulator()
        self.pP = Preprocessor()
    
        self.agent = Agent(self.iS, self.pP, True)
        
        input_var = self.agent.input_var

        a_n = T.ivector()       # Vector of actions
        r_n = T.fvector()       # Vector of rewards
        
        N = input_var.shape[0]
        
        prediction = self.agent.prediction
        loss = T.log(prediction[T.arange(N), a_n]).dot(r_n) / N
        
        params = lasagne.layers.get_all_params(self.agent.network, 
                                                    trainable=True)
        
        updates = lasagne.updates.rmsprop(loss, params, 
                                          learning_rate = self.learning_rate,
                                          rho = self.rho,
                                          epsilon = self.eps)
        self.prediction_fn = self.agent.pred_fn
        
        print("Compiling Training Function...")
        self.train_fn = theano.function([input_var, a_n, r_n], loss, updates=updates)
        print("Compiling Done!")
        
    def discount_rewards(self, r):
        discounted_r = np.zeros_like(r)
        running_add = 0.0
        for t in reversed(xrange(0, r.size)):
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
            
        #normalize
        discounted_r -= np.mean(discounted_r)
        discounted_r /= np.std(discounted_r)
        return discounted_r
        
    def train(self, epochs, batch_size):
        print("Start Training the PGLearner.")
        old_score = 0.0
        x_n, r_n, a_n = [],[],[]
        self.iS.clickPlay()
        time.sleep(3.5)
        self.agent.updateInput()
        for episode in range(epochs):
            start_time = time.time()
            while True:                
                x, action = self.agent.oneAction()
                self.agent.updateInput()
                score = self.agent.pP.getScore()
            
                x_n.append(x)
                r_n.append(score-old_score)
                a_n.append(action)
            
                if self.agent.pP.tryAgain() or self.agent.pP.highScore():
                    
                    x_s = np.vstack(x_n)
                    r_s = np.vstack(r_n)
                    a_s = np.vstack(a_n)
                    
                    x_n, r_n, a_n = [],[],[]
                    r_s[-1] = 0
                    rd_s = self.discount_rewards(r_s)
                    
                    self.train_fn(x_s,a_s,rd_s)
                    
                    print("Game {} of {} took {:.3f}s".format(
                    episode + 1, epochs, time.time() - start_time))
                    
                    if self.agent.pP.tryAgain():
                        self.iS.clickTryAgain()
                        time.sleep(3.5)
                        self.agent.updateInput()
                        break
                    elif self.agent.pP.highScore():
                        self.iS.enterInitials()
                        time.sleep(1.0)
                        self.iS.clickTryAgain()
                        time.sleep(3.5)
                        self.agent.updateInput()
                        break
            
            
            
            
            
            
def main():
    PGL = PGLearner(0.9, 0.0005, 0.9, 1e-6)
    PGL.train(800)
if __name__ == '__main__':
    main()
        