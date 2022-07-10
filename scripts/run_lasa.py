import numpy as np
from utils import *
import os
from elmap_driver import perform_elmap
import matplotlib.pyploy as plt

LAMBDA = 0.1
MU = 0.01

def run_lasa():
    lasa_names = ['Angle','BendedLine','CShape','DoubleBendedLine','GShape', \
                'heee','JShape','JShape_2','Khamesh','Leaf_1', \
                'Leaf_2','Line','LShape','NShape','PShape', \
                'RShape','Saeghe','Sharpc','Sine','Snake', \
                'Spoon','Sshape','Trapezoid','Worm','WShape', \
                'Zshape']
            
    weights = 'curvature'
    downsample = 'distance-based'
        
    for name in lasa_names:
        plt_fpath = '../pictures/lasa_reproductions/' + name + '/'
        
        try:
            os.makedirs(plt_fpath)
        except OSError:
            print ("Creation of the directory %s failed" % plt_fpath)
        else:
            pass
            
        ### SINGLE DEMONSTRATION
        
        [x, y] = get_lasa_trajn(name)
        traj = np.hstack((np.reshape(x, (len(x), 1)), np.reshape(y, (len(y), 1))))
        
        repro_single = perform_elmap(traj, n=100, weighting=weights, downsampling=downsample, lmbda=LAMBDA, mu=MU)
        
        fig = plt.figure()
        plt.plot(traj[:, 0], traj[:, 1], 'k', lw=5, alpha=0.3)
        plt.plot(repro_single[:, 0], repro_single[:, 1], 'r', lw=5)
        plt.savefig(plt_fpath + 'repro_single.png', dpi=300, bbox_inches='tight')
        plt.close('all')
        
        ### MULTIPLE DEMONSTRATIONS
        fig = plt.figure()
        demos = []
        demo_weights = []
        for i in range(7):
            [x, y] = get_lasa_trajn(name, i + 1)
            traj = np.hstack((np.reshape(x, (len(x), 1)), np.reshape(y, (len(y), 1))))
            demos.append(traj)
            demo_weights.append(calc_curv_weights(traj))
            
            plt.plot(traj[:, 0], traj[:, 1], 'k', lw=5, alpha=0.3)
            
        zip_data = zip_demos(demos)
        zip_weights = zip_demos(demo_weights)
        
        repro_multi = perform_elmap(zip_data, n=100, weighting='custom', downsampling=downsample, lmbda=LAMBDA, mu=MU, given_weights=zip_weights)
        plt.plot(repro_multi[:, 0], repro_multi[:, 1], 'r', lw=5)
        plt.savefig(plt_fpath + 'repro_multi.png', dpi=300, bbox_inches='tight')
        plt.close('all')
        
        ### MULTIPLE DEMONSTRATIONS AND CONSTRAINTS
        
        fig = plt.figure()
        
        indeces = [0, 50, 99]
        constants = [zip_data[0] * 1.3, zip_data[len(zip_data) // 2] * 1.3, zip_data[len(zip_data) - 1]]
        
        
        repro_constraints = perform_elmap(zip_data, n=100, weighting='custom', downsampling=downsample, inds=indeces, consts=constants, lmbda=0.00001, mu=0.00001, given_weights=zip_weights)
    
        for i in range(7):
            plt.plot(demos[i][:, 0], demos[i][:, 1], 'k', lw=5, alpha=0.3)
        plt.plot(repro_constraints[:, 0], repro_constraints[:, 1], 'r', lw=5)
        for i in range(len(indeces)):
            plt.plot(constants[i][0], constants[i][1], 'ko', ms=15, mew=5)
        
        plt.savefig(plt_fpath + 'repro_constraints.png', dpi=300, bbox_inches='tight')
        plt.close('all')
        
if __name__ == '__main__':
    run_lasa()