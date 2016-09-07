
import sys
sys.path.append("..\segmentation")

import matplotlib.pylab as plt
import numpy as np
from numpy import *
import time
from screenSegmentation import screenSegmentation
from sklearn.cluster import DBSCAN



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
        image = np.hstack((zeros((20,5)), image))
        image = np.hstack((pF, image))

        return image

    def getScore(self):
        db = DBSCAN(eps=0.3, min_samples=10).fit(self.scoreField)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_


        return score

def main():
    #plt.ion()

    tmp = np.loadtxt('digit_276.txt')
    db = DBSCAN(eps=0.3, min_samples=10).fit(tmp)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)


    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)

        xy = tmp[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)

        xy = tmp[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()

#if __name__ == '__main__':
#   main()

main()




