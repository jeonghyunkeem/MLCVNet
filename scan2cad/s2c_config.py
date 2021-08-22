import numpy as np
import sys
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from s2c_map import ID2NAME, NAME2CLASS

class Scan2CADDatasetConfig(object):
    def __init__(self):
        # dataset
        self.max_data_size = 15000
        self.chunk_size = 1000
        
        # max number
        self.max_num_point = 40000
        self.max_num_obj = 64
        self.ins_num_point = 2048

        # semantics
        self.num_class = 35
        self.num_heading_bin = 1
        self.num_size_cluster = 35
        self.sym2class = {"__SYM_NONE": 0, "__SYM_ROTATE_UP_2": 1, "__SYM_ROTATE_UP_4": 2, "__SYM_ROTATE_UP_INF": 3}

        # category mapping
        self.ShapenetIDToName = ID2NAME
        self.ShapenetNameToClass = NAME2CLASS
        self.ClassToName = {self.ShapenetNameToClass[t]:t for t in self.ShapenetNameToClass}

        # anchor
        mean_size_array = np.load(os.path.join(BASE_DIR, 'meta_data/s2c_means.npy'))
        self.mean_size_arr = mean_size_array[:, :3]
        self.class_total = mean_size_array[:, 3:4].astype(np.int64).squeeze(1).tolist()
        
    def ShapenetIDtoClass(self, cat_id):
        assert(cat_id in self.ShapenetIDToName)
        cat_name = self.ShapenetIDToName[cat_id]
        cat_cls = self.ShapenetNameToClass[cat_name]
        return cat_cls

    def class_summary(self):
        self.class_total = {cat_cls:self.class_total[cat_cls] for cat_cls in range(len(self.class_total))}
        class_total_summary = {k: v for k, v in sorted(self.class_total.items(), key=lambda item: item[1], reverse=True)}
        for i, key in enumerate(class_total_summary.keys()):
            print('{:2d}: {:12s} => {:4d}'.format(i, self.ClassToName[key], self.class_total[key]))

    def angle2class(self, angle):
        ''' Convert continuous angle to discrete class
            [optinal] also small regression number from  
            class center angle to current angle.
           
            angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
            return is class of int32 of 0,1,...,N-1 and a number such that
                class*(2pi/N) + number = angle
        '''
        num_class = self.num_heading_bin
        angle = angle%(2*np.pi)
        assert(angle>=0 and angle<=2*np.pi)
        angle_per_class = 2*np.pi/float(num_class)
        shifted_angle = (angle+angle_per_class/2)%(2*np.pi)
        class_id = int(shifted_angle/angle_per_class)
        residual_angle = shifted_angle - (class_id*angle_per_class+angle_per_class/2)
        return class_id, residual_angle

    def class2angle(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class '''
        num_class = self.num_heading_bin
        angle_per_class = 2*np.pi/float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format and angle>np.pi:
            angle = angle - 2*np.pi
        return angle

def rotate_aligned_boxes(input_boxes, rot_mat):    
    centers, lengths = input_boxes[:,0:3], input_boxes[:,3:6]    
    new_centers = np.dot(centers, np.transpose(rot_mat))
           
    dx, dy = lengths[:,0]/2.0, lengths[:,1]/2.0
    new_x = np.zeros((dx.shape[0], 4))
    new_y = np.zeros((dx.shape[0], 4))
    
    for i, crnr in enumerate([(-1,-1), (1, -1), (1, 1), (-1, 1)]):        
        crnrs = np.zeros((dx.shape[0], 3))
        crnrs[:,0] = crnr[0]*dx
        crnrs[:,1] = crnr[1]*dy
        crnrs = np.dot(crnrs, np.transpose(rot_mat))
        new_x[:,i] = crnrs[:,0]
        new_y[:,i] = crnrs[:,1]
    
    
    new_dx = 2.0*np.max(new_x, 1)
    new_dy = 2.0*np.max(new_y, 1)    
    new_lengths = np.stack((new_dx, new_dy, lengths[:,2]), axis=1)
                  
    return np.concatenate([new_centers, new_lengths], axis=1)

if __name__ == "__main__":
    Config = Scan2CADDatasetConfig()
    Config.class_summary()