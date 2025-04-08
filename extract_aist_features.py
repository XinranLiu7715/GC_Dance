import os
import numpy as np
import argparse
from features.kinetic import extract_kinetic_features
from features.manual_new import extract_manual_features
import pickle


from vis import SMPLX_Skeleton, SMPLSkeleton

import torch
import multiprocessing
import functools
import pdb

from pytorch3d.transforms import (RotateAxisAngle, 
                                  quaternion_multiply
                                  )
from pytorch3d.transforms import (
                                  quaternion_to_axis_angle, quaternion_apply)
from pytorch3d.transforms import (matrix_to_axis_angle,rotation_6d_to_matrix,
                                  axis_angle_to_quaternion
                                  )

from dataset.quaternion import ax_from_6v


def parse_opt():
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument(
        '--anno_dir',
        type=str,
        default='/mnt/welles/scratch/xu/xl/FineDance/test/motions_sliced/',
        help='input motion pkl.')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='test/fine/gt_motion/',
        help='output local dictionary that stores features.')
    parser.add_argument(
            "--eval_gt", 
            action="store_true", 
            help="Evaluate the GT feature.")
    opt = parser.parse_args()
    return opt


    

def process_dataset(root_pos, local_q):
        # to Tensor
        root_pos = torch.Tensor(root_pos)
        local_q = torch.Tensor(local_q)
        # to ax

        # 1 x 300 x 156
        bs, sq, c = local_q.shape
        local_q = local_q.reshape((bs, sq, -1, 3))
     
        # dataset comes y-up - rotate to z-up to standardize against the pretrain dataset
        
        #
        root_q = local_q[:, :, :1, :]  # sequence x 1 x 3
        root_q_quat = axis_angle_to_quaternion(root_q)
        rotation = torch.Tensor(
            [0.7071068, 0.7071068, 0, 0]
        )  # 90 degrees about the x axis
        root_q_quat = quaternion_multiply(rotation, root_q_quat)
        root_q = quaternion_to_axis_angle(root_q_quat)
        local_q[:, :, :1, :] = root_q
        # don't forget to rotate the root position too 😩
        pos_rotation = RotateAxisAngle(90, axis="X", degrees=True)
        root_pos = pos_rotation.transform_points(
            root_pos
        )  # basically (y, z) -> (-z, y), expressed as a rotation for readability

        

        return root_pos,local_q



def extract_feature(opt,seq_name,motion_dir,save_dir,gt=0,hand=False):

    # Parsing SMPL 24 joints.
    # Note here we calculate `transl` as `smpl_trans/smpl_scaling` for 
    # normalizing the motion in generic SMPL model scale.    
    
    path = os.path.join(motion_dir,seq_name)
    smpl_file = pickle.load(open(path, "rb"))
    hand = hand
    if gt ==1 :
        s, c = smpl_file.shape
        smpl_file = torch.from_numpy(smpl_file)
        smpl_file = smpl_file.reshape(1,s,c)
        _, model_out = torch.split(smpl_file, (4, smpl_file.shape[2] - 4), dim=2)
        model_x = model_out[:, :, :3]
        model_q = ax_from_6v(model_out[:, :, 3:].reshape(1, s, -1, 6))
        if opt.nfeats == 139:
            smpl = SMPLSkeleton()
            keypoints3d = smpl.forward(model_q, model_x)
            keypoints3d = keypoints3d.reshape(s,-1,3).numpy()
        else:
            smpl = SMPLX_Skeleton()
            b, s, nums, c_ = model_q.shape
            model_q = model_q.view(s, -1)
            model_x = model_x.view(-1, 3)
            keypoints3d = smpl.forward(model_q, model_x).numpy()
            if hand:
                keypoints3d = keypoints3d[:,25:,:]
            else:
                keypoints3d = keypoints3d[:,:22,:]
        #print(keypoints3d.shape)

    else:
        keypoints3d = smpl_file['full_pose']
        #keypoints3d = keypoints3d.cpu().numpy()
        if opt.nfeats == 319:
            if hand:
                keypoints3d = keypoints3d[:,25:,:]
            else:
                keypoints3d = keypoints3d[:,:22,:]
        
        #print(keypoints3d.shape)
        

    # test the body 
    roott = keypoints3d[:1, :1]  # the root
    keypoints3d = keypoints3d - roott  # Calculate relative offset with respect to root
    #features: 32 x 1

    if gt != 1:
        seq_name  =seq_name.split('_')[-2:]
        seq_name =seq_name[0]+'_'+seq_name[1]
    #features = extract_manual_features(keypoints3d)
    #np.save(os.path.join(save_dir, 'manual_features', seq_name.split('.pkl')[0]+"_manual.npy"), features)
    
    # keypoint3s: N x 24 x 3 
    # features: 72 x 1
    features = extract_kinetic_features(keypoints3d)
    np.save(os.path.join(save_dir, 'kinetic_features', seq_name.split('.pkl')[0]+"_kinetic.npy"), features)
    
    #print (seq_name, "is done")


if __name__ == '__main__':
    '''
    FLAGS = parse_opt()
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    os.makedirs(os.path.join(FLAGS.save_dir, 'kinetic_features'), exist_ok=True)
    os.makedirs(os.path.join(FLAGS.save_dir, 'manual_features'), exist_ok=True)
    seq_names = sorted(os.listdir(FLAGS.anno_dir))
    for name in seq_names:
        extract_feature(name,motion_dir=FLAGS.anno_dir)
    '''