import glob
import os
from functools import cmp_to_key
from pathlib import Path
from tempfile import TemporaryDirectory
import random
import pickle
import shutil

import numpy as np
import torch
from tqdm import tqdm
from eval.eval_beat import calc_ba_score
from eval.eval_pfc import calc_physical_score
from transformers import AutoTokenizer, CLIPModel
import json
import pdb
import math

from eval.eval_metrics import calc_and_save_feats, quantized_metrics




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



def test(model,opt):
    all_cond = []
    all_filenames = []
    all_text_fea = []
    music_dir = opt.data_path+'test_all/wav2clip_fea'
    music_basic_dir = opt.data_path+'test_all/Basic_music_fea/sftf'
    music_dir = '/mnt/fast/nobackup/scratch4weeks/xl01315/ICML/sampling/wav2clip_fea'
    music_basic_dir = '/mnt/fast/nobackup/scratch4weeks/xl01315/ICML/sampling/sftf'

    motion_save_dir = "test/motions"
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
        if len(cond_list) == 0:
            continue
        cond_list = torch.from_numpy(np.array(cond_list))
        text_list = torch.from_numpy(np.array(text_list))
        all_cond.append(cond_list)
        all_filenames.append(file_list)
        all_text_fea.append(text_list)
        print(idx,len(file_list))


    #print("Predict..")
    model.eval()
    #print("Generating dances")
    for i in range(len(all_cond)):
        data_tuple = None, all_cond[i], all_filenames[i], all_text_fea[i]
        render_count = -1
        model.render_sample(
            data_tuple, 
            "test", 
            "test/fine/renders/",
            render_count=render_count, 
            fk_out=motion_save_dir, 
            render=False,
        )

    #print("Done")

    gt_path = 'test/'
    cond_path = '/mnt/fast/nobackup/scratch4weeks/xl01315/dataset/test_all/wav_gt'
    pred_path = motion_save_dir
    gt_motion_path = opt.data_path+'test_all/motion_sliced_gt'

    
    hand = True
    mofea_gt_path = os.path.join(gt_path,'mofea_gt_52_hand')
    mofea_pred_path = os.path.join(gt_path,'mofea_pred_52_hand')
    #calc_and_save_feats(opt,mofea_gt_path,gt_motion_path,1,hand)
    #pred motion
    calc_and_save_feats(opt,mofea_pred_path,pred_path,0,hand)
    res_hand = quantized_metrics(mofea_pred_path, mofea_gt_path)

    hand = False
    mofea_gt_path = os.path.join(gt_path,'mofea_gt_52_body')
    mofea_pred_path = os.path.join(gt_path,'mofea_pred_52_body')
    #calc_and_save_feats(opt,mofea_gt_path,gt_motion_path,1,hand)
    #pred motion
    calc_and_save_feats(opt,mofea_pred_path,pred_path,0,hand)
    res_body = quantized_metrics(mofea_pred_path, mofea_gt_path)

    
    plc = calc_physical_score(pred_path)
    if math.isnan(plc):
        plc = 0
    align = calc_ba_score(cond_path,pred_path)
    torch.cuda.empty_cache()

    
    return res_hand,res_body,plc,align
    






