#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import os
import random
import pickle
from functools import cmp_to_key
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import random
import pickle
import shutil

import numpy as np
import torch
from tqdm import tqdm
import librosa
import librosa as lr
import soundfile as sf
import pdb

from args import test_opt
from train import GCdance

from eval.eval_beat import calc_ba_score
from eval.eval_pfc import calc_physical_score
from eval.eval_pbc import calc_physical_body_score
from eval.eval_metrics import calc_and_save_feats, quantized_metrics
from transformers import AutoTokenizer, CLIPModel
import json

def eval_metirc(opt):

    gt_path = 'tests/'
    pred_path = 'tests/motions'

    cond_path = 'tests/wav_gt'
    gt_motion_path = 'tests/motion_sliced_gt'
    
    pred_path = "tests/motions"
    


    cond_path = os.path.join(gt_path,'wav_gt')
    gt_motion_path = os.path.join(gt_path,'motion_sliced_gt')

    
    
    hand = True
    select = 'hand'
    mofea_gt_path = os.path.join('tests','mofea_gt_52_'+select)
    mofea_pred_path = os.path.join('tests','mofea_pred_52_'+select)
    calc_and_save_feats(opt,mofea_gt_path,gt_motion_path,1,hand)
    #pred motion
    calc_and_save_feats(opt,mofea_pred_path,pred_path,0,hand)
    hand_re = quantized_metrics(mofea_pred_path, mofea_gt_path)

    select = 'body'
    hand = False
    mofea_gt_path = os.path.join('tests','mofea_gt_52_'+select)
    mofea_pred_path = os.path.join('tests','mofea_pred_52_'+select)
    calc_and_save_feats(opt,mofea_gt_path,gt_motion_path,1,hand)
    #pred motion
    calc_and_save_feats(opt,mofea_pred_path,pred_path,0,hand)
    body_re = quantized_metrics(mofea_pred_path, mofea_gt_path)
    
    

    hand = None
    #calculate the beat aglin..
    bas = calc_ba_score(cond_path,pred_path)
    pfc = calc_physical_score(pred_path)
    pdb = calc_physical_body_score(pred_path)

    print('hand_re:',hand_re)
    print('body_re:',body_re)
    print('bas:',bas)
    print('pfc:',pfc)

    return

def move_gt(data_path):
    motion = sorted(glob.glob("tests/motions/*"))
    for i in motion:
        moname = os.path.basename(i)
        os.makedirs("tests/motion_sliced_gt/", exist_ok=True)
        sourename = os.path.join(data_path,"test_all/motion_gt",moname)
        target_name = "tests/motion_sliced_gt/"+moname
        shutil.copy(sourename, target_name)

    for i in motion:
        moname = os.path.basename(i)
        wavname = moname.replace('pkl','wav')
        os.makedirs("tests/wav_gt", exist_ok=True)
        sourename = os.path.join(data_path,"test_all/wav_gt",wavname)
        target_name = "tests/wav_gt/"+wavname
        shutil.copy(sourename, target_name)

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
test_list = ["063", "132", "143", "036", "098", "198", "012", "211", "193", "179", "065", "137", "161", "092", "037", "109", "204", "144"]
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
    music_fm_dir = os.path.join(opt.data_path,'test_all/wav2clip_fea')
    music_basic_dir = os.path.join(opt.data_path,'test_all/stft')
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    motion_save_dir = "tests/motions"
    feature_type = "baseline"
    json_file = opt.genre_json
    name_to_style = {}
    
    with open(json_file, 'r') as file:
        styles = json.load(file)
    name_to_style = {music['filename']: music['style2'] for music in styles}
    if opt.use_cached_features:             # default is false
        print("Using precomputed features")
        # all subdirectories
        dir_list = glob.glob(os.path.join(opt.feature_cache_dir, "*/"))
        for dir in dir_list:
            file_list = sorted(glob.glob(f"{dir}/*.wav"), key=stringintkey)
            juke_file_list = sorted(glob.glob(f"{dir}/*.npy"), key=stringintkey)
            assert len(file_list) == len(juke_file_list)
            # random chunk after sanity check
            rand_idx = random.randint(0, len(file_list) - sample_size)
            file_list = file_list[rand_idx : rand_idx + sample_size]
            juke_file_list = juke_file_list[rand_idx : rand_idx + sample_size]
            cond_list = [np.load(x) for x in juke_file_list]
            all_filenames.append(file_list)
            all_cond.append(torch.from_numpy(np.array(cond_list)))
    else:
        for songname in test_list:
            print('songname',songname)
            wav_file = os.path.join(opt.music_dir,songname+'.wav')
            songname = os.path.splitext(os.path.basename(wav_file))[0]
            #create temp folder (or use the cache folder if specified)
            wav_dir = os.path.join(opt.data_path,'test_all/wav_gt')
            all_file = sorted(glob.glob(f"{wav_dir}/*.wav"), key=stringintkey)
            file_list = []
            for f in all_file:
                if songname in f:
                    file_list.append(f)
            num = len(file_list)
            if num <= 15:
                sample_size = num
                rand_idx = 0
            else:
                sample_size = 15
                rand_idx = random.randint(0, len(file_list) - sample_size)

            cond_list = []
            text_list=[]
            filelist=[]


            # generate juke representations
            for idx, file in enumerate(tqdm(file_list)):
                if (not (rand_idx <= idx < rand_idx + sample_size)):
                    continue
                filename = os.path.basename(file)
                wav_fea_dir = os.path.join(music_fm_dir,file.split('/')[-1].replace('.wav','.pkl'))
                basic_fea_dir = os.path.join(music_basic_dir,file.split('/')[-1].replace('.wav','.npy'))
                reps_basic = np.load(basic_fea_dir).T
                reps_fm = pickle.load(open(wav_fea_dir, "rb")) .T
                reps =  np.concatenate((reps_fm, reps_basic), axis=-1)
                wav_name = os.path.join(file.split(feature_type)[0],'wavs_sliced',filename.split('.')[0]+'.wav')
                filelist.append(wav_name)
                save_name = filename.split('_')[0]
                genre = name_to_style[save_name]
                prompt1 = 'This is a '
                prompt2 = ' type of music.'
                y = [prompt1 + genre + prompt2]
                inputs = tokenizer(y, padding=True, return_tensors="pt")
                text_embeddings = clip_model.get_text_features(**inputs)

                # save reps
                if rand_idx <= idx < rand_idx + sample_size:
                    cond_list.append(reps)
                    text_list.append(text_embeddings[0].detach().numpy())


            cond_list = torch.from_numpy(np.array(cond_list))
            text_list = torch.from_numpy(np.array(text_list))
            all_cond.append(cond_list)
            all_filenames.append(filelist)
            all_text_fea.append(text_list)
            print(len(cond_list))
            print(len(text_list))

    model = GCdance(opt, opt.feature_type, opt.checkpoint)
    model.eval()
    
    print("Generating dances")
    for i in range(len(all_cond)):
        data_tuple = None, all_cond[i], all_filenames[i], all_text_fea[i]
        model.render_sample(
            data_tuple, 
            "test", 
            "test/renders/",
            render_count=-1, 
            fk_out=motion_save_dir, 
            mode="normal",  
            render=False,
        )
    print("Done")
    torch.cuda.empty_cache()


def process_once(opt):
    # delete all folder of /tests
    test_dir = "./test"
    if os.path.exists(test_dir):
        for name in os.listdir(test_dir):
            path = os.path.join(test_dir, name)
            if os.path.isdir(path):
                shutil.rmtree(path)
                print(f"🧹 Deleted folder: {path}")

    #test once
    test(opt)
    move_gt(opt.data_path)
    eval_metirc(opt)

if __name__ == "__main__":
    
    RUN_TIMES = 1
    
    opt = test_opt()
    if RUN_TIMES == 10:
        for i in range(RUN_TIMES):
            print(f"\n🔁 Running evaluation round {i+1}/{RUN_TIMES} ...")
            process_once(opt)
    else:
        process_once(opt)
        
