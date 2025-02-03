import numpy as np
import pickle 
import json
import argparse
import os
from essentia.standard import *
from  scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema
from extractor import FeatureExtractor
import torch
from dataset.quaternion import ax_to_6v, ax_from_6v
import matplotlib.pyplot as plt 

extractor = FeatureExtractor()
def extract_acoustic_feature(audio, sr):

    melspe_db = extractor.get_melspectrogram(audio, sr)
    
    mfcc = extractor.get_mfcc(melspe_db)
    mfcc_delta = extractor.get_mfcc_delta(mfcc)
    # mfcc_delta2 = get_mfcc_delta2(mfcc)

    audio_harmonic, audio_percussive = extractor.get_hpss(audio)
    # harmonic_melspe_db = get_harmonic_melspe_db(audio_harmonic, sr)
    # percussive_melspe_db = get_percussive_melspe_db(audio_percussive, sr)
    chroma_cqt = extractor.get_chroma_cqt(audio_harmonic, sr,  octave=7 if sr==30*512 else 5)
    # chroma_stft = extractor.get_chroma_stft(audio_harmonic, sr)

    onset_env = extractor.get_onset_strength(audio_percussive, sr)
    tempogram = extractor.get_tempogram(onset_env, sr)
    onset_beat = extractor.get_onset_beat(onset_env, sr)[0]
    # onset_tempo, onset_beat = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    # onset_beats.append(onset_beat)

    onset_env = onset_env.reshape(1, -1)

    feature = np.concatenate([
        # melspe_db,
        mfcc, # 20
        mfcc_delta, # 20
        # mfcc_delta2,
        # harmonic_melspe_db,
        # percussive_melspe_db,
        # chroma_stft,
        chroma_cqt, # 12
        onset_env, # 1
        onset_beat, # 1
        tempogram
    ], axis=0)

            # mfcc, #20
            # mfcc_delta, #20

            # chroma_cqt, #12
            # onset_env, # 1
            # onset_beat, #1

    feature = feature.transpose(1, 0)
    #print(f'acoustic feature -> {feature.shape}')

    return feature

def get_mb(cond_path,key, length=None):
    path = os.path.join(cond_path+'/'+key)
    sr = 512 * 30
    with open(path) as f:
        loader = essentia.standard.MonoLoader(filename=path, sampleRate=sr)
        audio = loader()
        audio = np.array(audio).T
        feature =  extract_acoustic_feature(audio, sr).tolist()
        
        if length is not None:
            beats = np.array(feature)[:, 53][:][:length]
        else:
            beats = np.array(feature)[:, 53]

        
        #audio_harmonic, audio_percussive = extractor.get_hpss(audio)
        #onset_env = extractor.get_onset_strength(audio_percussive, sr)
        #onset_beat = extractor.get_onset_beat(onset_env, sr)[0]


        beats = beats.astype(bool)
        beat_axis = np.arange(len(beats))
        beat_axis = beat_axis[beats]
        return beat_axis


def calc_db(keypoints, name=''):
    _,b = keypoints.shape
    keypoints = np.array(keypoints).reshape(-1, b//3, 3)
    keypoints = keypoints
    kinetic_vel = np.mean(np.sqrt(np.sum((keypoints[1:] - keypoints[:-1]) ** 2, axis=2)), axis=1)
    kinetic_vel = G(kinetic_vel, 5)
    motion_beats = argrelextrema(kinetic_vel, np.less)
    return motion_beats, len(kinetic_vel)


def BA(music_beats, motion_beats):
    ba = 0
    for bb in music_beats:
        if len(motion_beats[0]) > 0:
            ba +=  np.exp(-np.min((motion_beats[0] - bb)**2) / 2 / 9)
    return (ba / len(music_beats))

def calc_ba_score(cond_path,root,gt=0):
    
    ba_scores = []

    if gt == 0:
        for pkl in os.listdir(root):
            if os.path.isdir(os.path.join(root, pkl)):
                continue
            #joint3d = np.load(os.path.join(root, pkl), allow_pickle=True)['smpl_poses'][:, :]
            joint3d = np.load(os.path.join(root, pkl), allow_pickle=True)['full_pose'][:120, :]
            seq,_,_ = joint3d.shape
            joint3d = joint3d.reshape(seq,-1)
            dance_beats, length = calc_db(joint3d, pkl) 
            print(pkl)
            wave_name =pkl.replace("pkl", "wav").split('_')[-2:]
            wave_name =wave_name[0]+'_'+wave_name[1]
            music_beats = get_mb(cond_path,wave_name, length)
            ba_scores.append(BA(music_beats, dance_beats))

    else:
        for pkl in os.listdir(root):
            if os.path.isdir(os.path.join(root, pkl)):
                continue
            smpl_file = pickle.load(open(os.path.join(root, pkl), "rb"))
            s, c = smpl_file.shape
            smpl_file = torch.from_numpy(smpl_file)
            smpl_file = smpl_file.reshape(1,s,c)
            _, model_out = torch.split(smpl_file, (4, smpl_file.shape[2] - 4), dim=2)
            model_q = ax_from_6v(model_out[:, :, 3:].reshape(1, s, -1, 6))
            b, s, nums, c_ = model_q.shape
            joint3d = model_q.view(s, -1)
            dance_beats, length = calc_db(joint3d, pkl) 
            wave_name =pkl.replace("pkl", "wav").split('_')[-2:]
            wave_name =wave_name[0]+'_'+wave_name[1]
            music_beats = get_mb(cond_path,wave_name, length)
            ba_scores.append(BA(music_beats, dance_beats))

        
    return np.mean(ba_scores)




if __name__ == '__main__':



    print(calc_ba_score(cond_path,gt_root,pred_root))
    
