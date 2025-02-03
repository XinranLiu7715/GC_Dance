import os
import sys
import json
import random
import argparse
import essentia
import essentia.streaming
from essentia.standard import *
import librosa as lr
import numpy as np
import torch
import wav2clip

def parse_eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--music_dir', type=str, default='dataset/finedance/music_wav')
    parser.add_argument('--store_dir', type=str, default='dataset/train/music')
    parser.add_argument('--sampling_rate', type=int, default=15360*2/8)
    args = parser.parse_args()
    return args


args = parse_eval_args()

store_dir = args.store_dir

if not os.path.exists(args.store_dir):
    os.mkdir(args.store_dir)

def extract_stft(fpath):
    FPS = 30
    HOP_LENGTH = 512
    SR = FPS * HOP_LENGTH
    EPS = 1e-6
    data, _ = lr.load(fpath, sr=SR)
    n_fft = 2048  
    hop_length = 512  
    stft = lr.stft(y = data, n_fft=384, hop_length=hop_length)
    return stft.T

def extract_w2c(fpath):
    FPS = 30
    HOP_LENGTH = 512
    SR = FPS * HOP_LENGTH
    audio, sr = lr.load(fpath, sr=SR)
    model = wav2clip.get_model(frame_length=HOP_LENGTH, hop_length=HOP_LENGTH)
    embeddings = wav2clip.embed_audio(audio, model)[0]
    return embeddings.T

def make_music_dance_set(music_dir,store_dir):
    print('---------- Extract features from raw audio ----------')
    audio_fnames = sorted(os.listdir(music_dir))
    for audio_fname in audio_fnames:
        print(audio_fname)
        fpath = os.path.join(music_dir, audio_fname)
        stft = extract_stft(fpath)
        np.save(os.path.join(store_dir,'basic_fea',f'{audio_fname[:-4]}.npy'),stft)
        w2c = extract_w2c(fpath)
        np.save(os.path.join(store_dir,'wav2clip_fea',f'{audio_fname[:-4]}.npy'),w2c)


if __name__ == '__main__':
    make_music_dance_set(args.music_dir,args.store_dir) 
