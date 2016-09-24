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
        self.epsilon = epsilon
        
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
        
        updates = lasagne.updates.adadelta(loss, params, 
                                          learning_rate = self.learning_rate,
                                          rho = self.rho,
                                          epsilon = self.epsilon)

        self.prediction_fn = self.agent.pred_fn
        
        print("Compiling Training Function...")
        self.train_fn = theano.function([input_var, a_n, r_n],
                                        [], 
                                        updates=updates,
                                        allow_input_downcast=True)
        print("Compiling Done!")
        
    def train(self, epochs, batch_size):
        print("Start Training the PGLearner.")

        r_n, a_n = [],[]
        self.iS.clickPlay()
        time.sleep(3.5)
        self.agent.updateInput()
        rd_s = np.empty((0,1))
        x_n = np.empty((0,4,20,20))
        
        for episode in range(epochs):
            start_time = time.time()
            eps = 1#0.5 + 0.5*(episode/(0.9*epochs))

            old_score = 0.0
            old_cleared = 0
            old_height = 0
            n_blocks_old = 0
            r_sum = 0
            while True:                
                x, action = self.agent.oneAction(eps)
                self.agent.updateInput()
                score = self.agent.pP.getScore()
                height = self.agent.pP.getHighestLine()
                cleared = self.agent.pP.getLinesCleared()
                n_blocks = self.agent.pP.getNBlocksInPlayField()
                
                if score == -1:
                    s = 0
                else:
                    s = ((score-old_score)/100)
                    
                if cleared == -1:
                    c = 0
                else:
                    c = (cleared-old_cleared)
                    
                h = (height-old_height)
                n = (n_blocks-n_blocks_old)
        
                if (-h != c and h < 0) or height < 4:
                    h = 0
                    
                conc = 0.0
                
                if n > 0:
                    conc = float(n_blocks)/(height*9)
                    conc = 2*(conc-0.33)
        
                c = 2.5*c
                s = 0.5*s
                reward = c + conc + s
            
                x_n = np.append(x_n, x, 0)
                a_n.append(action)
                r_n.append(reward)
            
                r_sum += reward    
            
                if self.agent.pP.isMenuOpen():
                    t = time.time() - start_time
                    r_n[-1] = -0.5
                    r_sum -=c
                    r_sum -= 0.5
                    print("Game {} of {} took {:.3f}s and reached a score of {}".format(
                            episode + 1, epochs, t, r_sum))
                    self.log(t, r_sum, old_score, old_cleared)
                    
                    rd_tmp = self.discount_rewards(np.vstack(r_n))
                    rd_s = np.concatenate((rd_s, rd_tmp))
                    
                    r_n = []
                    
                    if np.mod(episode + 1, 1) == 0:
                        x_s = x_n
                        a_s = np.vstack(a_n)
                    
                        a_n = []

                        a_s = a_s.reshape(a_s.shape[0],)
                        rd_s = rd_s.reshape(rd_s.shape[0],)
                    
                        shuf = np.arange(x_s.shape[0])
                        np.random.shuffle(shuf)
                    
                        for k in range(np.floor_divide(x_s.shape[0], batch_size)):
                            it1 = k*batch_size
                            it2 = (k+1)*batch_size
                            self.train_fn(x_s[shuf[it1:it2],:,:,:],
                                          a_s[it1:it2],rd_s[it1:it2])
                            
                        rd_s = np.empty((0,1))
                        x_n = np.empty((0,4,20,20))
                    
                    self.agent.resetInput()                  
                    time.sleep(0.5)
                    
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
                    
                old_score = score
                old_cleared = cleared
                old_height = height
                    
        print("Saving Model to File...")
        self.agent.saveParams('trained_model1.npz')
        print("End Training Program!")
            
    def discount_rewards(self, r):
        discounted_r = np.zeros_like(r, np.float32)
        running_add = 0.0
        for t in reversed(xrange(0, r.size)):
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
            
        return discounted_r
        
    def log(self, time, reward, score, cleared):
        with open("log_file.txt", "a") as myfile:
            myfile.writelines(str(time) + ', ' + str(reward) + ', ' + str(score) + ', ' + str(cleared) +'\n')
            
            
def main():
    PGL = PGLearner(0.99, 0.001, 0.9, 1e-6, False)
    PGL.train(100,30)
#==============================================================================
#     x = np.zeros((1,4,20,20),np.int8)
#     r = np.array([1])
#     a = np.array([3])
#     a = a.reshape(a.shape[0],)
#     r = r.reshape(r.shape[0],)
#     r.astype(np.float32)
#     print 'eta: ',PGL.learning_rate
#     print 'rho: ',PGL.rho
#     print 'eps: ',PGL.epsilon
#     print PGL.prediction_fn(x)
#     PGL.train_fn(x,a,r)
#     print PGL.prediction_fn(x)
#==============================================================================
    
    
if __name__ == '__main__':
    main()
        