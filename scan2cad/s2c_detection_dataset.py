# Jeonghyun Kim, UVR KAIST @jeonghyunct.kaist.ac.kr

""" Scan-to-CAD dataset

Partial Scan <-> CAD Point cloud from Scan2CAD annotation

Notation:
    Parital Scan:
        v: (40000, 3)
            partial scan w/ region clipped
        vs: (2048, 3)
            instance partial scan 
        vc: (2038, 3)
            canonical instance parital scan
    
    CAD Point cloud:
        cad_v: (2048, 3)
            sampled point cloud from ShapeNetCore
        alignments: (10)
            translation(3), rotation(4), scale(3) from Scan2CAD alignment annotation

    Semantics:
        sem_cls: (1)
            CAD object category
        symmetry: (1)
            symmetry class of CAD object
    
"""
import os, sys, time
import numpy as np
import torch
import quaternion
from torch.utils.data import Dataset
from glob import glob
import h5py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
HOME_DIR = os.path.dirname(ROOT_DIR)
DATA_DIR = os.path.join(HOME_DIR, 'Dataset/Scan2CAD')
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util
from s2c_config import Scan2CADDatasetConfig, rotate_aligned_boxes

DC = Scan2CADDatasetConfig()
MAX_NUM_POINT = DC.max_num_point
MAX_INS_POINT = DC.ins_num_point
CHUNK_SIZE = DC.chunk_size
MAX_NUM_OBJ = 256

DATA_SPLIT = ['train', 'val', 'test']
NOT_CARED_IDS = np.array([-1])

def from_q_to_6d(q):
    q = np.quaternion(q[0], q[1], q[2], q[3])
    mat = quaternion.as_rotation_matrix(q)  # 3x3
    rep6d = mat[:, 0:2].transpose().reshape(-1, 6)   # 6
    return rep6d

class Scan2CADDetectionDataset(Dataset):
    def __init__(self, split='train', num_points=20000, augment=False, use_color=False, use_height=False):
        super().__init__()
        # read data
        self.data_path = os.path.join(DATA_DIR, 'detection')
        scene_path_all = list(set([os.path.basename(x)[:12] \
            for x in os.listdir(self.data_path) if x.startswith('scene')]))
        self.error_scan = os.path.join(BASE_DIR, 'meta_data', 'error_scan.txt')
        if split == 'all':
            self.scene_list = scene_path_all
            num_scans = len(self.scene_list)
        elif split in ['train', 'val', 'test']:
            split_file = os.path.join(BASE_DIR, 'meta_data', 'scan2cad_{}.txt'.format(split))
            with open(split_file, 'r') as f:
                self.scene_list = f.read().splitlines()
            num_scans = len(self.scene_list)
            self.scene_list = [scene for scene in self.scene_list if scene in scene_path_all]
        if self.error_scan in self.scene_list:
            self.scene_list.remove(self.error_scan)
        print('kept {} scans out of {}'.format(len(self.scene_list), num_scans))
        
        self.num_points = num_points
        self.use_color = use_color        
        self.use_height = use_height
        self.augment = augment

    def __len__(self):
        return len(self.scene_list)

    def __getitem__(self, index: int):
        id_scan = self.scene_list[index]
        assert(id_scan not in self.error_scan)
        id_scan_path = os.path.join(self.data_path, id_scan)
        point_cloud = np.load(os.path.join(id_scan_path, '{}.npy'.format('point_cloud')))
        ins_vert    = np.load(os.path.join(id_scan_path, '{}.npy'.format('ins_vert'))).squeeze(1)
        ins_bbox    = np.load(os.path.join(id_scan_path, '{}.npy'.format('bbox')))

        points      = point_cloud           # (N, 3)
        center      = ins_bbox[:, 0:3]      # (B, 10)
        bbox_length = ins_bbox[:, 3:6]      # (B, 3)
        sem_cls     = ins_bbox[:, 6:7]      # (B, 1)
        symmetry    = ins_bbox[:, 7:8]      # (B, 1)

        K = center.shape[0]

        # LABELS 
        if points.shape[0] > self.num_points:
            points, choices = pc_util.random_sampling(points, self.num_points, return_choices=True)
            ins_vert = ins_vert[choices]
        elif points.shape[0] < self.num_points:
            print('false data')
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6), dtype=np.float32)
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ), dtype=np.int64)
        # target_center   = np.zeros((MAX_NUM_OBJ, 3), dtype=np.float32)
        # target_rot_q    = np.zeros((MAX_NUM_OBJ, 4), dtype=np.float32)
        # target_rot_6d   = np.zeros((MAX_NUM_OBJ, 6), dtype=np.float32)
        # target_scale    = np.zeros((MAX_NUM_OBJ, 3), dtype=np.float32)
        target_sem_cls  = np.zeros((MAX_NUM_OBJ,), dtype=np.int64)
        target_sym      = np.zeros((MAX_NUM_OBJ,), dtype=np.int64)
        target_size_classes = np.zeros((MAX_NUM_OBJ,))
        target_size_residuals = np.zeros((MAX_NUM_OBJ, 3))

        # target_center[:K]       = center[:,0:3]
        # target_rot_q[:K]    = alignments[:,3:7]
        # for k in range(K):
        #     target_rot_6d[k]    = from_q_to_6d(alignments[k,3:7])
        # target_scale[:K]        = alignments[:,7:10]
        target_sem_cls[:K]      = sem_cls.squeeze(1)
        target_sym[:K]          = symmetry.squeeze(1)

        target_bboxes[:K, 0:3] = center
        target_bboxes[:K, 3:6] = bbox_length
        target_bboxes_mask[:K] = 1

        # ------------------------------- DATA AUGMENTATION ------------------------------        
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                points[:,0] = -1 * points[:,0]
                target_bboxes[:,0] = -1 * target_bboxes[:,0]                
                
            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                points[:,1] = -1 * points[:,1]
                target_bboxes[:,1] = -1 * target_bboxes[:,1]                                
            
            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
            rot_mat = pc_util.rotz(rot_angle)
            points[:,0:3] = np.dot(points[:,0:3], np.transpose(rot_mat))
            target_bboxes = rotate_aligned_boxes(target_bboxes, rot_mat)

        target_center = target_bboxes[:, 0:3]
        # ====== GENERATE VOTES ====== 
        # compute votes *AFTER* augmentation
        # NOTE: i_ins: (1,B) not (0,B-1) 
        point_votes = np.zeros([self.num_points, 3])
        point_votes_mask = np.zeros(self.num_points)
        for i_ins in np.unique(ins_vert):
            i_ins -= 1
            if target_sem_cls[i_ins] in NOT_CARED_IDS or i_ins < 0:
                continue
            ind = np.where(ins_vert == i_ins+1)[0]
            x = points[ind, :3]
            point_votes[ind, :] = x - target_center[i_ins]
            point_votes_mask[ind] = 1.0
        point_votes = np.tile(point_votes, (1, 3))
        
        target_size_classes[:K] = target_sem_cls[:K]
        target_size_residuals[:K, :3] =\
            target_bboxes[:K, 3:6] - DC.mean_size_arr[target_sem_cls[:K], :]

        # ====== LABELS ======
        label = {}
        label['point_clouds'] = points.astype(np.float32)
        label['center_label'] = target_center.astype(np.float32)
        label['heading_class_label'] = np.zeros((MAX_NUM_OBJ,)).astype(np.int64)
        label['heading_residual_label'] = np.zeros((MAX_NUM_OBJ,)).astype(np.float32)
        label['size_class_label'] = target_size_classes.astype(np.int64)
        label['size_residual_label'] = target_size_residuals.astype(np.float32)
        label['sem_cls_label'] = target_sem_cls.astype(np.int64)
        label['box_label_mask'] = target_bboxes_mask.astype(np.float32)
        label['vote_label'] = point_votes.astype(np.float32)
        label['vote_label_mask'] = point_votes_mask.astype(np.int64)
        label['scan_idx'] = np.array(index).astype(np.int64)

        return label

if __name__ == "__main__":
    D = Scan2CADDetectionDataset(split='train')
    pcd, label = D[1]
    print(pcd.shape, label.keys())