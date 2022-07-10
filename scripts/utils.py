import numpy as np
import h5py
import matplotlib.pyplot as plt

#extract demo from the lasa_dataset.h5 file
def get_lasa_trajn(shape_name, n=1):
    # ask user for the file which the playback is for
    # filename = raw_input('Enter the filename of the .h5 demo: ')
    # open the file
    filename = '../h5 files/lasa_dataset.h5'
    hf = h5py.File(filename, 'r')
    # navigate to necessary data and store in numpy arrays
    shape = hf.get(shape_name)
    demo = shape.get('demo' + str(n))
    pos_info = demo.get('pos')
    pos_data = np.array(pos_info)
    y_data = np.delete(pos_data, 0, 1)
    x_data = np.delete(pos_data, 1, 1)
    # close out file
    hf.close()
    return [x_data, y_data]
  
#read data from a real-world demo taken in the PeARL Lab
def read_3D_h5(fname):
    #ask user for the file which the playback is for
    #filename = raw_input('Enter the filename of the .h5 demo: ')
    #open the file
    hf = h5py.File(fname, 'r')
    #navigate to necessary data and store in numpy arrays
    demo = hf.get('demo1')
    tf_info = demo.get('tf_info')
    pos_info = tf_info.get('pos_rot_data')
    pos_data = np.array(pos_info)
    
    x = pos_data[0, :]
    y = pos_data[1, :]
    z = pos_data[2, :]
    #close out file
    hf.close()
    return [x, y, z]
    
#calculate curvature of points to use as weights for elastic map  
def calc_curv_weights(data):
    nodeWeights = [0]
    for i in range(1, len(data) - 1):
        weight = np.linalg.norm(data[i - 1] + data[i + 1] - 2 * data[i])
        nodeWeights.append(weight)
    nodeWeights.append(0)
    nodeWeights[0] = nodeWeights[1]
    nodeWeights[-1] = nodeWeights[-2]
    nodeWeights = np.array(nodeWeights).reshape((len(data), 1))
    return nodeWeights
    
#calculate jerk of points to use as weights for elastic map
def calc_jerk_weights(data):
    nodeWeights = []
    for i in range(0, len(data)):
        if(i < 2 or i>len(data)-3):
            weight = 0
        else:
            weight = np.linalg.norm(data[i - 2] + 2*data[i - 1] - 2 * data[i+1] - data[i+2])
        nodeWeights.append(weight)
    nodeWeights[0] = nodeWeights[2]
    nodeWeights[1] = nodeWeights[2]
    nodeWeights[-1] = nodeWeights[-3]
    nodeWeights[-2] = nodeWeights[-3]
    nodeWeights = np.array(nodeWeights).reshape((len(data), 1))
    return nodeWeights
    
#"zip" a list of demonstrations into a time-aligned stack for elastic map to model
def zip_demos(d):
    nd = len(d)
    (n_pts, n_dims) = np.shape(d[0])
    fd = np.zeros((n_pts * nd, n_dims))
    for i in range(n_pts):
        for id in range(nd):
            fd[(i * nd) + id][:] = d[id][i][:]
    return fd
    
#calculate total jerk of a trajectory
def calc_jerk(traj):
    (n_pts, n_dims) = np.shape(traj)
    ttl = 0.
    for i in range(2, n_pts - 2):
        ttl += np.linalg.norm(traj[i - 2] + 2*traj[i - 1] - 2 * traj[i+1] - traj[i+2])
    return ttl
    
#calculate angular similarity for 2 trajectories of different lengths
def align_ang_sim(exp_data, num_data):
    exp = exp_data - exp_data[0]
    num = num_data - num_data[0]
    
    #assume exp data is always larger
    new_exp = [exp[0]]
    indeces = [i for i in range(1, len(exp) - 1)]
    for i in range(1, len(num) - 1):
        best_ind = indeces[0]
        best_val = np.linalg.norm(exp[indeces[0]] - num[i])
        for j in indeces:
            val = np.linalg.norm(exp[j] - num[i])
            if val < best_val:
                best_val = val
                best_ind = j
        new_exp.append(exp[best_ind])
        indeces.remove(best_ind)
        
    new_exp.append(exp[-1])
    new_exp = np.array(new_exp)
    return angular_similarity(new_exp, num)    

#calculate angular similarity for 2 trajectories
def angular_similarity(exp_data, num_data):
    (n_points, n_dims) = np.shape(exp_data)
    if not (np.shape(exp_data) == np.shape(num_data)):
        print('Array dims must match!')
        
    sum = 0.
    for i in range(n_points - 1):
        v1 = exp_data[i + 1] - exp_data[i]
        v2 = num_data[i + 1] - num_data[i]
        #calc cosine similarity
        costheta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        #calc angular distance
        ang_dist = np.arccos(costheta) / np.pi if not np.isnan(costheta) else 0
        sum = sum + ang_dist
    return sum / n_points