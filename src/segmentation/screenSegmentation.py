from PIL import ImageGrab
import matplotlib.pylab as plt
import numpy as np
from numpy.lib.stride_tricks import as_strided
import os
import sys


class screenSegmentation():

    def __init__(self):
        path, _ = os.path.split(os.path.abspath(sys.modules[screenSegmentation.__module__].__file__))

        self.topleft = np.load(path + '\\topleftcorner.npy' )
        self.bottomright = np.load(path + '\\bottomright.npy' )

        I = self.screenGrab()
        
        conv = self.ssd(I.astype(float), self.topleft.astype(float))
        
        if np.min(conv) <= 0.01:
            self.oy = np.unravel_index(np.argmin(conv),conv.shape)[0]
            self.ox = np.unravel_index(np.argmin(conv),conv.shape)[1]
        else:
            raise Exception('Tetris not found')
            
        conv = self.ssd(I.astype(float), self.bottomright.astype(float))
        
        if np.min(conv) <= 0.01:
            self.ey = np.unravel_index(np.argmin(conv),conv.shape)[0]
            self.ex = np.unravel_index(np.argmin(conv),conv.shape)[1]
        else:
            raise Exception('Tetris not found')

    def getGame(self):
        I = self.screenGrab()
        return self.section(self.ox, self.oy, self.ex, self.ey, I)
        
    def getCurrentState(self):
        game = self.getGame()
        playfield = self.section(274, 47, 274+269, 47+528, game)
        holdfield = self.section(92, 110, 92+120, 110+78, game)
        nextfield = self.section(603, 109, 603+124, 109+214, game)
        scorefield = self.section(94, 390, 94+115, 390+27, game)
        
        return game, playfield, holdfield, nextfield, scorefield
 
    def screenGrab(self):
        im = ImageGrab.grab().convert('L')
        arr = np.fromiter(iter(im.getdata()), np.uint8)
        arr.resize(im.height, im.width)
        return arr
        
    def section(self, sx, sy, ex, ey, arr):
        return arr[sy:ey, sx:ex]
        
    def ssd( self, input_image, template):
        window_size = template.shape
        y = as_strided(input_image,
                        shape=(input_image.shape[0] - window_size[0] + 1,
                               input_image.shape[1] - window_size[1] + 1,) +
                              window_size,
                        strides=input_image.strides * 2)
        ssd = np.einsum('ijkl,kl->ij', y, template)
        ssd *= - 2
        ssd += np.einsum('ijkl, ijkl->ij', y, y)
        ssd += np.einsum('ij, ij', template, template)
    
        return ssd
        
    def getGameOrigin(self):
        return self.ox, self.oy

    
def main():
    segmentation = screenSegmentation()
    game, playfield, holdfield, nextfield, scorefield = segmentation.getCurrentState()
    plt.imshow(game, cmap='Greys_r')
    plt.show()

 
if __name__ == '__main__':
    main()