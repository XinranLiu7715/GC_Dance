import glob
import os
from functools import cmp_to_key
from pathlib import Path
from tempfile import TemporaryDirectory
#import jukemirlib
import wav2clip
import pickle
import numpy as np
import torch
from tqdm import tqdm
import librosa
import librosa as lr
import soundfile as sf
from args import test_opt


def slice_audio(audio_file, stride, length, out_dir):
    # stride, length in seconds
    audio, sr = lr.load(audio_file, sr=None)
    file_name = os.path.splitext(os.path.basename(audio_file))[0]
    start_idx = 0
    idx = 0
    window = int(length * sr)
    stride_step = int(stride * sr)
    while start_idx <= len(audio) - window:
        audio_slice = audio[start_idx : start_idx + window]
        sf.write(f"{out_dir}/{file_name}_slice{idx}.wav", audio_slice, sr)
        full_out_dir = 'test_all/wav_gt'
        sf.write(f"{full_out_dir}/{file_name}_slice{idx}.wav", audio_slice, sr)
        start_idx += stride_step
        idx += 1
    return idx


def slice_motion(motion_file, stride = 60, length = 120):
    motion = np.load(motion_file)
    file_name = os.path.splitext(os.path.basename(motion_file))[0]
    # normalize root position
    start_idx = 0
    window = 120
    stride_step = 60
    idx = 0
    l = len(motion)
    # slice until done or until matching audio slices
    while start_idx <= l - window:
        motion_seq = motion[start_idx : start_idx + window]
        full_out_dir = 'test_all/motion_gt'
        pickle.dump(motion_seq, open(f"{full_out_dir}/{file_name}_slice{idx}.pkl", "wb"))
        start_idx += stride_step
        idx += 1
    return idx

def extract(fpath):
    FPS = 30
    HOP_LENGTH = 512
    SR = FPS * HOP_LENGTH
    EPS = 1e-6

    data, _ = librosa.load(fpath, sr=SR)
    envelope = librosa.onset.onset_strength(y=data, sr=SR)  # (seq_len,)
    mfcc = librosa.feature.mfcc(y=data, sr=SR, n_mfcc=20).T  # (seq_len, 20)
    chroma = librosa.feature.chroma_cens(
        y=data, sr=SR, hop_length=HOP_LENGTH, n_chroma=12
    ).T  # (seq_len, 12)

    peak_idxs = librosa.onset.onset_detect(
        onset_envelope=envelope.flatten(), sr=SR, hop_length=HOP_LENGTH
    )
    peak_onehot = np.zeros_like(envelope, dtype=np.float32)
    peak_onehot[peak_idxs] = 1.0  # (seq_len,)

    start_bpm = lr.beat.tempo(y=lr.load(fpath)[0])[0]

    tempo, beat_idxs = librosa.beat.beat_track(
        onset_envelope=envelope,
        sr=SR,
        hop_length=HOP_LENGTH,
        start_bpm=start_bpm,
        tightness=100,
    )
    beat_onehot = np.zeros_like(envelope, dtype=np.float32)
    beat_onehot[beat_idxs] = 1.0  # (seq_len,)

    audio_feature = np.concatenate(
        [envelope[:, None], mfcc, chroma, peak_onehot[:, None], beat_onehot[:, None]],
        axis=-1,
    )

    # chop to ensure exact shape
    audio_feature = audio_feature[:4 * FPS]
    return audio_feature


def extract_stft(fpath,out_dir):
    FPS = 30
    HOP_LENGTH = 512
    SR = FPS * HOP_LENGTH
    data, _ = librosa.load(fpath, sr=SR)
    hop_length = 512 
    stft = librosa.stft(y = data, n_fft=384, hop_length=hop_length)[:,:120]
    print(stft.shape)
    name = fpath.split('/')[-1].replace('.wav','.npy')
    save_path = os.path.join(out_dir,name)
    np.save(save_path,stft)
    return stft

def extract_w2c(fpath,out_dir):
    FPS = 30
    HOP_LENGTH = 512
    SR = FPS * HOP_LENGTH
    audio, sr = lr.load(fpath, sr=SR)
    model = wav2clip.get_model(frame_length=HOP_LENGTH, hop_length=HOP_LENGTH)
    embeddings = wav2clip.embed_audio(audio, model)[0]
    print(embeddings.shape)
    name = fpath.split('/')[-1].replace('.wav','.pkl')
    pickle.dump(embeddings, open(f"{out_dir}/{name}", "wb"))
    return embeddings


def juke_extract(fpath):
    FPS = 30
    LAYER = 66
    audio = jukemirlib.load_audio(fpath)
    reps = jukemirlib.extract(audio, layers=[LAYER], downsample_target_rate=FPS)
    return reps

# sort filenames that look like songname_slice{number}.ext
key_func = lambda x: int(os.path.splitext(x)[0].split("_")[-1].split("slice")[-1])
test_list = ["063", "132", "143", "036", "098", "198", "130", "012", "211", "193", "179", "065", "137", "161", "092", "120", "037", "109", "204", "144"]



def stringintcmp_(a, b):
    aa, bb = "".join(a.split("_")[:-1]), "".join(b.split("_")[:-1])
    ka, kb = key_func(a), key_func(b)
    if aa < bb:
        return -1
    if aa > bb:
        return 1
    if ka < kb:
        return -1
    if ka > kb:
        return 1
    return 0

stringintkey = cmp_to_key(stringintcmp_)
stride_ = 60/30

def Slice(opt):
    sample_length = opt.out_length
    sample_size = int(sample_length / stride_)
    temp_dir_list = []
    music_dir = 'data/finedance/music_wav'
    motion_dir = 'data/train/motion_fea319'
    stft_out = 'test_all/stft'
    wac2clip_out = 'test_all/wav2clip_fea'

    print("Computing features for input music")
    for filename in test_list:
        wav_file =  os.path.join(music_dir,filename+'.wav')
        temp_dir = TemporaryDirectory()
        temp_dir_list.append(temp_dir)
        dirname = temp_dir.name
        print(f"Slicing {wav_file}")
        slice_audio(wav_file, 2, 4, dirname)
        motion_file = os.path.join(motion_dir,filename+'.npy')
        slice_motion(motion_file)
        file_list = sorted(glob.glob(f"{dirname}/*.wav"), key=stringintkey)
        for file in file_list:
            extract_stft(file,stft_out)
            extract_w2c(file,wac2clip_out)
            
            

  





if __name__ == "__main__":
    if not os.path.exists('test_all'):
        os.mkdir('test_all')
    if not os.path.exists('test_all/wav_gt'):
        os.mkdir('test_all/wav_gt')
    if not os.path.exists('test_all/wav2clip_fea'):
        os.mkdir('test_all/wav2clip_fea')
    if not os.path.exists('test_all/stft'):
        os.mkdir('test_all/stft')
    if not os.path.exists('test_all/motion_gt'):
        os.mkdir('test_all/motion_gt')

    opt = test_opt()
    Slice(opt)
