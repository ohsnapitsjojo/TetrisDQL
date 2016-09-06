import sys
sys.path.append("..\segmentation")
import matplotlib.pylab as plt
import numpy as np
from numpy import *
import time
from screenSegmentation import screenSegmentation


class Preprocessor():

    def __init__(self):
        self.sS = screenSegmentation()
        self.game, self.playField, self.holdField, self.nextField, self.scoreField = self.sS.getCurrentState()

    def update(self):
        self.game, self.playField, self.holdField, self.nextField, self.scoreField = self.sS.getCurrentState()


    def preprocessPlayField(self):
        x,y = 16, 16
        step = 26
        pooledPF = self.playField[x::step, y::step]
        pooledPF = pooledPF > 0
        pooledPF = pooledPF*254

        return pooledPF

    def preprocessNextField(self):
        x, y = 16, 16
        step = 26
        pooledNF = self.nextField[x::step, y::step]
        pooledNF = pooledNF > 36
        pooledNF = pooledNF * 254

        return pooledNF

    def preprocessHoldField(self):
        x, y = 16, 16
        step = 26
        pooledNF = self.holdField[x::step, y::step]
        pooledNF = pooledNF > 36
        pooledNF = pooledNF * 254

        return pooledNF

    def preprocess(self):
        pF = self.preprocessPlayField()
        nF = self.preprocessNextField()
        hF = self.preprocessHoldField()

        image = np.hstack((hF, zeros((3, 1))))
        image = np.vstack((image, zeros((9,5))))
        image = np.vstack((image, nF ))
        image = np.hstack((zeros((20,1)), image))
        image = np.hstack((pF, image))

        return image

def main():
    pP = Preprocessor()
    plt.ion()

    x, y = 16, 16
    step = 26
    while(1):
        tmp = pP.preprocess()
        plt.imshow(tmp, cmap='Greys_r', interpolation='none')
        plt.pause(0.0001)
        pP.update()

if __name__ == '__main__':
    main()





