import argparse
import glob
import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
from vis import SMPLX_Skeleton, SMPLSkeleton
from dataset.quaternion import ax_from_6v
import random

def calculate_acceleration(joint_data, dt):
    velocity = (joint_data[1:] - joint_data[:-1]) / dt
    acceleration = (velocity[1:] - velocity[:-1]) / dt
    normalized_acceleration = np.linalg.norm(acceleration, axis=-1)
    if normalized_acceleration.max() != 0:
        normalized_acceleration /= normalized_acceleration.max()
    return normalized_acceleration

def calc_physical_body_score(dir,gt=0):
    scores = []
    names = []
    accelerations = []
    up_dir = 2  # z is up
    flat_dirs = [i for i in range(3) if i != up_dir]
    cube_dirs = [i for i in range(3)]
    DT = 1 / 30

    it = glob.glob(os.path.join(dir, "*.pkl"))
    if len(it) > 1000:
        it = random.sample(it, 1000)
    for pkl in tqdm(it):
        info = pickle.load(open(pkl, "rb"))
        if gt == 0:
            joint3d = info["full_pose"]
        else:
            smpl = SMPLX_Skeleton()
            smpl_file = info
            s, c = smpl_file.shape
            smpl_file = torch.from_numpy(smpl_file)
            smpl_file = smpl_file.reshape(1,s,c)
            _, model_out = torch.split(smpl_file, (4, smpl_file.shape[2] - 4), dim=2)
            model_x = model_out[:, :, :3]
            model_q = ax_from_6v(model_out[:, :, 3:].reshape(1, s, -1, 6))
            b, s, nums, c_ = model_q.shape
            model_q = model_q.view(s, -1)
            model_x = model_x.view(-1, 3)
            keypoints3d = smpl.forward(model_q, model_x).numpy()
            joint3d = keypoints3d

            

        root_v = (joint3d[1:, 0, :] - joint3d[:-1, 0, :]) / DT  
        root_a = (root_v[1:] - root_v[:-1]) / DT 
        root_a[:, up_dir] = np.maximum(root_a[:, up_dir], 0)
        root_a = np.linalg.norm(root_a, axis=-1)  # (S-2,)
        scaling = root_a.max()
        root_a /= scaling

        joint_indices = [13, 14, 12]
        all_accelerations = []
        for idx in joint_indices:
            acc = calculate_acceleration(joint3d[:, idx, :], DT)
            all_accelerations.append(acc)

        foot_idx = [7, 10, 8, 11]
        body_idx = [20, 22, 21, 23, 15]

        feet = joint3d[:, foot_idx] 
        foot_v = np.linalg.norm(
            feet[2:, :, flat_dirs] - feet[1:-1, :, flat_dirs], axis=-1
        )
        foot_mins = np.zeros((len(foot_v), 2))
        foot_mins[:, 0] = np.minimum(foot_v[:, 0], foot_v[:, 1])
        foot_mins[:, 1] = np.minimum(foot_v[:, 2], foot_v[:, 3])

        body= joint3d[:, body_idx]
        body_v = np.linalg.norm(
            body[2:, :, cube_dirs] - body[1:-1, :, cube_dirs], axis=-1
        )
        body_mins = np.zeros((len(body_v), 3))
        body_mins[:, 0] = np.minimum(body_v[:, 0], body_v[:, 1])
        body_mins[:, 1] = np.minimum(body_v[:, 2], body_v[:, 3])
        body_mins[:, 2] = body_v[:,4]

        body_loss = (
            0.01 * body_mins[:,0] * all_accelerations[0] + 0.01 * body_mins[:,1] * all_accelerations[1] + 0.1 * body_mins[:, 2] * all_accelerations[2] - foot_mins[:, 0] * foot_mins[:, 1] * root_a 
        ) 

        body_loss = body_loss.mean()
        scores.append(body_loss)
        names.append(pkl)
        accelerations.append(foot_mins[:, 0].mean())

    out = np.mean(scores) * 10000
    print(f"{dir} has a mean PFC of {out}")




if __name__ == "__main__":
    print('gt')
    gt = 1
    calc_physical_body_score('tests/motion_sliced_gt',gt)
    print('pred')
    calc_physical_body_score('tests/motions')
