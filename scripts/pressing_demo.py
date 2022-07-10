import numpy as np
from utils import *
import os
from elmap_driver import perform_elmap
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def pressing_3D_test():
    [x, y, z] = read_3D_h5('../h5 files/recorded_demo Tue Apr 20 09_53_21 2021.h5')


    traj = np.hstack((np.reshape(x, (len(x), 1)), np.reshape(y, (len(y), 1)), np.reshape(z, (len(z), 1))))

    og_indices = [1580, 3060]
    new_indices = [33, 66]
    constants = [traj[og_indices[0]], traj[og_indices[1]]]
    for i in range(len(og_indices)):
    	constants[i][2] = constants[i][2] - 0.005
        
    repro = perform_elmap(traj, n=100, weighting='curvature', downsampling='distance-based', inds=new_indices, consts=constants, lmbda=1e-7, mu=1e-6)
    print(np.shape(repro))
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    demo, = ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'k', lw=3)
    rep, = ax.plot(repro[:, 0], repro[:, 1], repro[:, 2], 'r', lw=3)
    for i in range(len(og_indices)):
    	cst, = ax.plot(constants[i][0], constants[i][1], constants[i][2], 'k.', ms=12, mew=3)
    
    plt.legend((demo, rep, cst), ('Demonstration', 'Reproduction', 'Constraints'), fontsize='x-large')
    
    plt.show()
	
if __name__ == '__main__':
    pressing_3D_test()