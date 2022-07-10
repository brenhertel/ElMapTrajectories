import sys
sys.path.insert(1, 'elastic-maps')

from elmap_EM import *
from MapGeometry import *
from matlab_like_funcs import *
from rect2DMap import *

import numpy as np
from utils import *
from downsampling import *

def perform_elmap(data, n=100, weighting='uniform', downsampling='naive', inds=[], consts=[], lmbda=0.1, mu=0.01, given_weights=[], given_downsample=[]):
    #calculate weights if needed
    if weighting == 'uniform':
        weights = np.ones((len(data),1))
    elif weighting == 'curvature':
        weights = calc_curv_weights(data)
    elif weighting == 'jerk':
        weights = calc_jerk_weights(data)
    elif weighting == 'custom':
        weights = given_weights
    else:
        print('Weighting method unrecognized! Cannot continue...')
        return []
        
    #perform downsampling if needed
    if downsampling == 'naive':
        nodes = downsample_traj(data, n)
    elif downsampling == 'distance-based':
        nodes = db_downsample(data, n)
    elif downsampling == 'douglas-peucker':
        nodes = DouglasPeuckerPoints(data, n)
    elif downsampling == 'custom':
        nodes = given_downsample
    else:
        print('Downsampling method unrecognized! Cannot continue...')
        return []
        
    #incorporate constraints if any
    for i in range(len(inds)):
        nodes[inds[i]] = consts[i]
        data = np.insert(data, 0, consts[i], axis=0)
        weights = np.insert(weights, 0, np.size(data), axis=0)
        
    #solve for optimized map
    
    #initialize objects
    map = rect2DMap(n, 1)
    map.init(data, _type='random')
    map.mapped = nodes
    #perform EM and return result
    EM(map, data, constStretching = lmbda, constBending = mu, weights=weights)
    nodes = map.getMappedCoordinates()
    return nodes
    
    
def test():
    import matplotlib.pyplot as plt
    
    x = np.linspace(0, 10)
    x = np.reshape(x, (len(x), 1))
    y1 = np.sin(x) + 0.05 * np.random.normal(size=np.shape(x))
    y2 = 0.9 * np.sin(x) + 0.05 * np.random.normal(size=np.shape(x))
    y3 = 1.1 * np.sin(x) + 0.05 * np.random.normal(size=np.shape(x))
    traj1 = np.hstack((x, y1))
    traj2 = np.hstack((x, y2))
    traj3 = np.hstack((x, y3))
    print(np.shape(np.random.normal(np.shape(x))))
    print(np.shape(y1))
    print(np.shape(traj1))
    data = zip_demos([traj1, traj2, traj3])
    
    #repro = perform_elmap(data, n=20, weighting='uniform', downsampling='naive', lmbda=0.1, mu=0.01)
    
    repro = perform_elmap(data, n=20, weighting='uniform', downsampling='naive', inds=[0, -1], consts=[[0, 0], [10, 0]], lmbda=0.01, mu=0.001)
    
    plt.figure()
    plt.plot(x, y1, 'k', lw=5)
    plt.plot(x, y2, 'k', lw=5)
    demo, = plt.plot(x, y3, 'k', lw=5)
    elmap, = plt.plot(repro[:, 0], repro[:, 1], 'r', lw=5)
    plt.plot(0, 0, 'k.', ms=20)
    cst, = plt.plot(10, 0, 'k.', ms=20)
    plt.legend((demo, elmap, cst), ('Demonstrations', 'Elastic Maps', 'Constraints'), fontsize='x-large')
    plt.show()
    
    
if __name__ == '__main__':
    test()