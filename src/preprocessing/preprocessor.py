
import sys
sys.path.append("..\segmentation")

import matplotlib.pylab as plt
import numpy as np

from screenSegmentation import screenSegmentation
from sklearn.cluster import DBSCAN
import math
import time

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

        image = np.hstack((hF, np.zeros((3, 1))))
        image = np.vstack((image, np.zeros((9,5))))
        image = np.vstack((image, nF ))
        image = np.hstack((np.zeros((20,5)), image))
        image = np.hstack((pF, image))

        return image

    def getScore(self, plot='False'):
        binarySF = self.scoreField
        binarySF = binarySF > 50
        points = [index for index, x in np.ndenumerate(binarySF) if x == 1 ]
        if not points:
            return 'Empty score field.'
        _, nCluster, nPoints= dbscan(points,plot)
        score = 0

        for i in range(1, nCluster+1):
            digit = self.getDigit(nPoints[i-1])
            if digit == -1:
                return -1
                
            score = score + digit*math.pow(10, nCluster-i)
            
        return score

    def getDigit(self, n):
        if n == 81:
            return 0
        if n == 34:
            return 1
        if n == 71:
            return 2
        if n == 78:
            return 3
        if n == 64:
            return 4
        if n == 74:
            return 5
        if n == 69:
            return 6
        if n > 45 and n < 58:
            return 7
        if n == 92:
            return 8
        if n == 68:
            return 9
        
        return -1
    
    def readScoreRT(self):
        nScore = 0
        oScore = 0   
        while(1):    
            self.update()
            nScore = self.getScore()
            if oScore != nScore:
                #print nScore
                pass
            oScore = nScore
            
    def isMenuOpen(self):
        if self.game[40,40] != 224:
            return True
        else:
            return False
    
    def tryAgain(self):
        if self.game[418,470] == 89:
            return True
        else:
            return False
    
    def highScore():
        pass
            
def dbscan(points, plot=False):
    if not points:
        return 0, 0, 0
        
    
    X = np.fliplr(np.asarray(points))
    db = DBSCAN(eps=1.5, min_samples=5).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # Black removed and is used for noise instead.
    i = 1
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    

    if plot == True:
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = 'k'
        
            class_member_mask = (labels == k)
        
            xy = X[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=14)
        
            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=6)    
        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.show() 
   
    scoreNPoints = np.zeros(n_clusters_)


    for i in range(n_clusters_):
        scoreNPoints[i] = np.sum(labels == i)  
    scoreNPoints = [x for x in scoreNPoints if x >15]
    n_clusters_ = len(scoreNPoints)

    points = dict()
    center = np.zeros(n_clusters_)    
    
    for n in range(n_clusters_):
        points[n] = [point for idx, point in enumerate(X) if labels[idx] == n ]
        tmp = np.asarray(points[n])
        center[n] = np.average(tmp[:,0])
    
    sortIdx = np.argsort(center)
    
    scoreNPoints = [scoreNPoints[sortIdx[i]] for i in range(n_clusters_)]
    #print scoreNPoints
    return labels, n_clusters_, scoreNPoints
    
def main():
    #plt.ion()
    #score = np.loadtxt('316.txt')
    #score = score > 50
    #points = [index for index, x in np.ndenumerate(score) if x == 1 ]
    #p = Preprocessor()
    #print p.getScore()
    #plt.imshow(score, interpolation='none',cmap='Greys_r')
    pass
    
#if __name__ == '__main__':
#   main()

main()




