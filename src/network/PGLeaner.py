import sys
sys.path.append("..\preprocessing")
sys.path.append("..\inputSimulator")

import time

import numpy as np
import theano
import theano.tensor as T

import lasagne

from preprocessor import Preprocessor
from inputSimulator import InputSimulator
from agent import Agent

import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

class PGLearner:
    
    def __init__(self, gamma, learning_rate, rho, epsilon, load_network):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.rho = rho
        self.eps = epsilon
        
        self.iS = InputSimulator()
        self.pP = Preprocessor()
    
        self.agent = Agent(self.iS, self.pP, load_network)
        
        input_var = self.agent.input_var

        a_n = T.ivector()       # Vector of actions
        r_n = T.fvector()       # Vector of rewards
        
        N = input_var.shape[0]
        
        prediction = self.agent.prediction
        loss = -T.log(prediction[T.arange(N), a_n]).dot(r_n) / N
        
        params = lasagne.layers.get_all_params(self.agent.network, 
                                                    trainable=True)
        
        updates = lasagne.updates.rmsprop(loss, params, 
                                          learning_rate = self.learning_rate,
                                          rho = self.rho,
                                          epsilon = self.eps)
        self.prediction_fn = self.agent.pred_fn
        
        print("Compiling Training Function...")
        self.train_fn = theano.function([input_var, a_n, r_n],
                                        [], 
                                        updates=updates,
                                        allow_input_downcast=True)
        print("Compiling Done!")
        
    def train(self, epochs, batch_size):
        print("Start Training the PGLearner.")
        old_score = 0.0
        old_cleared = 0
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
                h = self.agent.pP.getHighestLine()
                cleared = self.agent.pP.getLinesCleared()
                
                if score == -1:
                    s = 0
                else:
                    s = ((score-old_score)/100)
                    
                if cleared == -1:
                    c = 0
                else:
                    c = (cleared-old_cleared)
        
                reward = s - h/20 + c/2
            
                x_n.append(x)
                a_n.append(action)
                r_n.append(reward)
                
                old_score = score
                old_cleared = cleared
            
                if self.agent.pP.isMenuOpen():
                    t = time.time() - start_time
                    x_s = np.vstack(x_n)
                    r_s = np.vstack(r_n)
                    a_s = np.vstack(a_n)
                    
                    x_n, r_n, a_n = [],[],[]
                    r_s[-1] = 0
                    r_sum = np.sum(r_s)
                    rd_s = self.discount_rewards(r_s)

                    a_s = a_s.reshape(a_s.shape[0],)
                    rd_s = rd_s.reshape(rd_s.shape[0],)
                    
                    #shuf = np.arange(x_s.shape[0])
                    #np.random.shuffle(shuf)
                    
                    #for k in range(np.floor_divide(x_s.shape[0], batch_size)):
                     #   it1 = k*batch_size
                      #  it2 = (k+1)*batch_size
                       # self.train_fn(x_s[shuf[it1:it2],:,:,:],
                        #              a_s[it1:it2],rd_s[it1:it2])
                    self.train_fn(x_s,a_s,rd_s)

                    print("Game {} of {} took {:.3f}s and reached a Score of {}".format(
                    episode + 1, epochs, t, r_sum))
                    
                    self.agent.resetInput()
                    self.log(t, r_sum)
                    
                    time.sleep(2)
                    
                    if self.agent.pP.tryAgain():
                        self.iS.clickTryAgain()
                        time.sleep(3.5)
                        self.agent.updateInput()
                        break
                    else:
                        self.iS.enterInitials()
                        time.sleep(1.0)
                        self.iS.clickTryAgain()
                        time.sleep(3.5)
                        self.agent.updateInput()
                        break
                    
        print("Saving Model to File...")
        self.agent.saveParams('trained_model1.npz')
        print("End Training Program!")
            
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
        
    def log(self, time, reward):
        with open("log_file.txt", "a") as myfile:
            myfile.writelines(str(time) + ', ' + str(reward) + '\n')
            
            
def main():
    PGL = PGLearner(0.99, 0.01, 0.9, 1e-6, False)
    PGL.train(200,5)
    
if __name__ == '__main__':
    main()
        