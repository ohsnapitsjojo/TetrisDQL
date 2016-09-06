import sys
sys.path.append("..\segmentation")

import matplotlib.pylab as plt
import numpy as np
from numpy import *

from screenSegmentation import screenSegmentation


class PreProcessor():

    def __init__(self):
        self.sS = screenSegmentation()
        self.game, self.playField, self.holdField, self.nextField, self.scoreField = self.sS.getCurrentState()

    def preprocessPlayField(self):
        x,y = 4, 4
        pooledPF = self.playField[x:12:, y:12:]


        return pooledPF


# def main():
#     pP = PreProcessor()
#     tmp = pP.preprocessPlayField()
#
#     plt.imshow(tmp, cmap='Greys_r')
#     plt.show()
#if __name__ == '__main__':
#    main()



x = range(100)
x = reshape(x, (10, 10))
y = x[1:3,0:3]
print y
print x

