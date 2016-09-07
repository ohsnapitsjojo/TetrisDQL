import sys
sys.path.append("..\preprocessing")
sys.path.append("..\inputSimulator")

import time

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

import lasagne

from network import build_network
from preprocessor import Preprocessor
from inputSimulator import InputSimulator


class Agent:
    
    def __init__(self, inputSimulator, preprocessor, pretrained):
        self.iS = inputSimulator
        self.pP = preprocessor
        self.input_var = T.tensor4('input')
        self.network = build_network(7, self.input_var, (20,20))
        
        self.prediction = lasagne.layers.get_output(self.network, deterministic=True)
        self.x = np.zeros((1,4,20,20),np.int8)
        
        if pretrained:
            self.loadParams()
        
        print("Compiling network")
        self.pred_fn = theano.function([self.input_var], self.prediction)
        print("Compiling done")
        
    def performAction(self, probabilities):
        # 0: None
        # 1: Up
        # 2: Left
        # 3: Right
        # 4: Down
        # 5: Space
        # 6: Hold
        
        action = np.argmax(probabilities)
        
        print 'Probabilities: ', probabilities
        if action == 0:
            print("No action.")
        elif action == 1:
            print("Up.")
            self.iS.up()
        elif action == 2:
            print("Left")
            self.iS.left()
        elif action == 3:
            print("Right")
            self.iS.right()
        elif action == 4:
            print("Down")
            self.iS.down()
        elif action == 5:
            print("Space")
            self.iS.space()
        elif action == 6:
            print("Hold")
            self.iS.c()
            
        return action
            
    def updateInput(self):
        self.x[0,3,:,:] = self.x[0,2,:,:]
        self.x[0,2,:,:] = self.x[0,1,:,:]
        self.x[0,1,:,:] = self.x[0,0,:,:]
        self.pP.update()
        self.x[0,0,:,:] = self.pP.preprocess()
        
        
    def play(self):
        self.updateInput()
        for i in range(200):
            self.updateInput()
            self.performAction(self.pred_fn(self.x))   
            
    def oneAction(self):
        self.updateInput()
        action = self.performAction(self.pred_fn(self.x))
        return action
            
    def loadParams(self):
        print("Loading paramters...")
        with np.load('trained_model.npz') as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(self.network, param_values)
        print("Loading Done!")
            
    def saveParams(self, string):
        print("Saving Model...")
        np.savez(string, lasagne.layers.get_all_param_values(self.network))
        print("Saving Done!")

def main():
    iS = InputSimulator()
    pP = Preprocessor()
    
    agent = Agent(iS, pP, True)
    print("Starting the game")
    iS.clickPlay()
    time.sleep(3.1)
    print("Agent is playing now")
    agent.play()

if __name__ == '__main__':
    main()