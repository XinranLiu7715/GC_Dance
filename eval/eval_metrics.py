import numpy as np
import pickle 
import glob
from scipy import linalg
import pdb
from extract_aist_features import extract_feature


from pytorch3d.transforms import (RotateAxisAngle, quaternion_to_axis_angle,
                                  quaternion_multiply, quaternion_apply,matrix_to_axis_angle,rotation_6d_to_matrix,
                                  axis_angle_to_quaternion
                                  )


# kinetic, manual
import os

def normalize(feat, feat2):
    mean = feat.mean(axis=0)
    std = feat.std(axis=0)
    return (feat - mean) / (std + 1e-10), (feat2 - mean) / (std + 1e-10)


def quantized_metrics(predicted_pkl_root, gt_pkl_root):

    pred_features_k = []
    #pred_features_m = []
    gt_freatures_k = []
    #gt_freatures_m = []

    

    pred_features_k = [np.load(os.path.join(predicted_pkl_root, 'kinetic_features', pkl)) for pkl in os.listdir(os.path.join(predicted_pkl_root, 'kinetic_features'))]
    #pred_features_m = [np.load(os.path.join(predicted_pkl_root, 'manual_features', pkl)) for pkl in os.listdir(os.path.join(predicted_pkl_root, 'manual_features'))]

    gt_freatures_k = [np.load(os.path.join(gt_pkl_root, 'kinetic_features', pkl)) for pkl in os.listdir(os.path.join(gt_pkl_root, 'kinetic_features'))]
    #gt_freatures_m = [np.load(os.path.join(gt_pkl_root, 'manual_features', pkl)) for pkl in os.listdir(os.path.join(gt_pkl_root, 'manual_features'))]
    
    pred_features_k = np.stack(pred_features_k)  # Nx72 p40
    #pred_features_m = np.stack(pred_features_m) # Nx32

    gt_freatures_k = np.stack(gt_freatures_k) # N' x 72 N' >> N
    #gt_freatures_m = np.stack(gt_freatures_m) # 

#   T x 24 x 3 --> 72
# T x72 -->32 
   
    gt_freatures_k, pred_features_k = normalize(gt_freatures_k, pred_features_k)
    #gt_freatures_m, pred_features_m = normalize(gt_freatures_m, pred_features_m) 
    

    '''
    print(pred_features_k.mean(axis=0))
    print(pred_features_m.mean(axis=0))
    print(pred_features_k.std(axis=0))
    print(pred_features_m.std(axis=0))
    '''

    #print('Calculating metrics')
    fid_k = calc_fid(pred_features_k, gt_freatures_k)
    #fid_m = calc_fid(pred_features_m, gt_freatures_m)

    div_k_gt = calculate_avg_distance(gt_freatures_k)
    #div_m_gt = calculate_avg_distance(gt_freatures_m)
    div_k = calculate_avg_distance(pred_features_k)
    #div_m = calculate_avg_distance(pred_features_m)

    if isinstance(fid_k, complex):
        fid_k = fid_k.real
    #if isinstance(fid_m, complex):
    #    fid_m = fid_m.real

    
    #metrics = {'fid_k': fid_k, 'fid_m': fid_m, 'div_k_gt': div_k_gt, 'div_m_gt': div_m_gt, 'div_k': div_k, 'div_m' : div_m}
    metrics = {'fid_k': fid_k, 'div_k_gt': div_k_gt,  'div_k': div_k }
    return metrics
  



def calc_fid(kps_gen, kps_gt):

    print(kps_gen.shape)
    print(kps_gt.shape)

    # kps_gen = kps_gen[:20, :]

    mu_gen = np.mean(kps_gen, axis=0)
    sigma_gen = np.cov(kps_gen, rowvar=False)

    mu_gt = np.mean(kps_gt, axis=0)
    sigma_gt = np.cov(kps_gt, rowvar=False)\

    mu1,mu2,sigma1,sigma2 = mu_gen, mu_gt, sigma_gen, sigma_gt

    diff = mu1 - mu2
    eps = 1e-5
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calc_diversity(feats):
    feat_array = np.array(feats)
    n, c = feat_array.shape
    diff = np.array([feat_array] * n) - feat_array.reshape(n, 1, c)
    return np.sqrt(np.sum(diff**2, axis=2)).sum() / n / (n-1)

def calculate_avg_distance(feature_list, mean=None, std=None):
    feature_list = np.stack(feature_list)
    n = feature_list.shape[0]
    # normalize the scale
    if (mean is not None) and (std is not None):
        feature_list = (feature_list - mean) / std
    dist = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist += np.linalg.norm(feature_list[i] - feature_list[j])
    dist /= (n * n - n) / 2
    return dist

def calc_and_save_feats(opt, save_dir, motion_dir,gt=0,hand=False):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'kinetic_features'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'manual_features'), exist_ok=True)
    seq_names = sorted(os.listdir(motion_dir))
    for name in seq_names:
        extract_feature(opt,name,motion_dir,save_dir,gt,hand)


if __name__ == '__main__':


    gt_root = 'test/GT'
    pred_root = 'test'
    
    #print('Calculating metrics')
    print(quantized_metrics(pred_root, gt_root))
    
