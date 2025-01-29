import glob
import os
from functools import cmp_to_key
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import random
import pickle

# import jukemirlib
import numpy as np
import torch
from tqdm import tqdm
import librosa
import librosa as lr
import soundfile as sf
import pdb

from args import FineDance_parse_test_opt
from train_seq import EDGE
# from data.audio_extraction.jukebox_features import extract as juke_extract
from eval.eval_beat import calc_ba_score
from eval.eval_pfc import calc_physical_score
from eval.eval_metrics import calc_and_save_feats, quantized_metrics
from transformers import AutoTokenizer, CLIPModel
import json
import pdb
import math

def eval_metirc(opt):
    if opt.test_10:
        gt_path = '/mnt/fast/nobackup/scratch4weeks/xl01315/dataset/test_10'
    else:
        gt_path = '/mnt/fast/nobackup/scratch4weeks/xl01315/dataset/test_all'
        
    pred_path = "tests/motions"
    
    if opt.type == 1:
        hand = True
        select = 'hand'
    elif opt.type == 2:
        select = 'body'
        hand = False
    else:
        hand = None
        select = ''
    cond_path = os.path.join(gt_path,'wav_gt')
    gt_motion_path = os.path.join(gt_path,'motion_sliced_gt')
    mofea_gt_path = os.path.join('tests','mofea_gt_52_'+select)
    mofea_pred_path = os.path.join('tests','mofea_pred_52_'+select)
    if hand == None:
        #calculate the beat aglin..
        print(calc_ba_score(cond_path,pred_path))
        #print(calc_ba_score(cond_path,gt_motion_path,1))

        #calculate the beat PFC..
        print(calc_physical_score(pred_path))
        #print(calc_physical_score(gt_motion_path,1))
    else:
        #calculate the fid and divt.
        #GT motion 
        #calc_and_save_feats(opt,mofea_gt_path,gt_motion_path,1,hand)
        #pred motion
        calc_and_save_feats(opt,mofea_pred_path,pred_path,0,hand)
        print(quantized_metrics(mofea_pred_path, mofea_gt_path))

    return




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

# sort filenames that look like songname_slice{number}.ext
key_func = lambda x: int(os.path.splitext(x)[0].split("_")[-1].split("slice")[-1])
test_list = ["063", "132", "143", "036", "098", "198", "012", "211", "193", "179", "065", "137", "161", "092",  "037", "109", "204", "144"]

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

def test(opt):
    all_cond = []
    all_filenames = []
    all_text_fea = []
    music_dir = opt.data_path+'test_all/wav2clip_fea'
    music_basic_dir = opt.data_path+'test_all/Basic_music_fea/sftf'

    motion_save_dir = "test_term/motion"
    feature_type = "baseline"
    json_file = opt.genre_json
    name_to_style = {}
    
    wav_fea =  sorted(glob.glob(os.path.join(music_dir,'*.pkl')), key=stringintkey)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    with open(json_file, 'r') as file:
        styles = json.load(file)
    name_to_style = {music['filename']: music['style2'] for music in styles}
    for idx in test_list:
        cond_list = []
        file_list= []
        text_list=[]
        for wavs in wav_fea:
            if idx not in wavs:
                continue
            filename = os.path.basename(wavs)
            wav_basic = os.path.join(music_basic_dir,filename).split('.')[0]+'.npy'
            reps = pickle.load(open(wavs, "rb"))[0]
            reps_basic = np.load(wav_basic).T
            reps =  np.concatenate((reps, reps_basic), axis=-1)
            cond_list.append(reps)
            wav_name = os.path.join(wavs.split(feature_type)[0],'wavs_sliced',filename.split('.')[0]+'.wav')
            file_list.append(wav_name)
            save_name = filename.split('_')[0]
            genre = name_to_style[save_name]
            prompt1 = 'This is a '
            prompt2 = ' type of music.'
            y = [prompt1 + genre + prompt2]
            inputs = tokenizer(y, padding=True, return_tensors="pt")
            text_embeddings = clip_model.get_text_features(**inputs)
            text_list.append(text_embeddings[0].detach().numpy())
        
        cond_list = torch.from_numpy(np.array(cond_list))
        text_list = torch.from_numpy(np.array(text_list))
        all_cond.append(cond_list)
        all_filenames.append(file_list)
        all_text_fea.append(text_list)
        print(idx,len(file_list))

        
    model = EDGE(opt, opt.feature_type, opt.checkpoint)
    model.eval()

    # directory for optionally saving the dances for eval


    print("Generating dances")
    for i in range(len(all_cond)):
        data_tuple = None, all_cond[i], all_filenames[i], all_text_fea[i]
        model.render_sample(
            data_tuple, 
            "test", 
            "renders/test",
            render_count=-1, 
            fk_out=motion_save_dir, 
            mode="normal",  
            render=True,
        )


    print("Done")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    opt = test_opt()
    test(opt)
    #eval_metirc(opt)

