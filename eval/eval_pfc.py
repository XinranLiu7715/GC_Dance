import argparse
import glob
import os
import pickle
from tempfile import TemporaryDirectory
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs
import torch
from dataset.quaternion import ax_from_6v
from vis import process_dataset
import random
from vis import SMPLX_Skeleton, SMPLSkeleton


def calc_physical_score(dir,gt=0):
    scores = []
    names = []
    accelerations = []
    up_dir = 2  # z is up
    flat_dirs = [i for i in range(3) if i != up_dir]
    DT = 1 / 30
    it = glob.glob(os.path.join(dir, "*.pkl"))
    if len(it) > 1000:
        it = random.sample(it, 1000)
    for pkl in tqdm(it):
        smpl_file = pickle.load(open(pkl, "rb"))
        if gt == 0:
            joint3d = smpl_file['full_pose'][:,:22,:]

        else:
            smpl = SMPLX_Skeleton()
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
            joint3d = keypoints3d[:,:22,:]


        root_v = (joint3d[1:, 0, :] - joint3d[:-1, 0, :]) / DT  # root velocity (S-1, 3)
        root_a = (root_v[1:] - root_v[:-1]) / DT  # (S-2, 3) root accelerations
        # clamp the up-direction of root acceleration
        root_a[:, up_dir] = np.maximum(root_a[:, up_dir], 0)  # (S-2, 3)
        # l2 norm
        root_a = np.linalg.norm(root_a, axis=-1)  # (S-2,)
        scaling = root_a.max()
        root_a /= scaling

        foot_idx = [7, 10, 8, 11]
        feet = joint3d[:, foot_idx]  # foot positions (S, 4, 3)
        foot_v = np.linalg.norm(
            feet[2:, :, flat_dirs] - feet[1:-1, :, flat_dirs], axis=-1
        )  # (S-2, 4) horizontal velocity
        foot_mins = np.zeros((len(foot_v), 2))
        foot_mins[:, 0] = np.minimum(foot_v[:, 0], foot_v[:, 1])
        foot_mins[:, 1] = np.minimum(foot_v[:, 2], foot_v[:, 3])

        foot_loss = (
            foot_mins[:, 0] * foot_mins[:, 1] * root_a
        )  # min leftv * min rightv * root_a (S-2,)
        foot_loss = foot_loss.mean()
        scores.append(foot_loss)
        names.append(pkl)
        accelerations.append(foot_mins[:, 0].mean())

    out = np.mean(scores) * 10000
    return out



    



def parse_eval_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motion_path",
        type=str,
        default="test/wild_motion",
        #default="test/pred_motion_sliced",
        help="Where to load saved motions",
    )
    parser.add_argument(
        "--motion_savepath",
        type=str,
        default="test/GT/motion_full",
        help="Where to load saved motions",
    )
    parser.add_argument(
        "--eval_gt", 
        action="store_true", 
        help="Evaluate the GT feature.")
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_eval_opt()
    path = 'test/aist/motions_sliced'
    calc_physical_score(path,1)