import numpy as np
from munkres import Munkres
from scipy.spatial.distance import cdist


def cluster_similarity(U1, U2):

    assert U1.shape[1] == U2.shape[1]

    cost = cdist(U1.T, U2.T, "hamming")
    indices = Munkres().compute(cost.copy())
    p1, p2 = zip(*indices)

    return 1 - np.mean(cost[p1, p2])
