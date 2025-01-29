import torch
from torch.utils import data
import numpy as np
import os
from tqdm import tqdm
import json
import pdb
import pickle
import sys
import librosa as lr
import soundfile as sf
from tempfile import TemporaryDirectory

from transformers import AutoTokenizer, CLIPModel

sys.path.insert(0,'.')

SMPL_JOINTS_FLIP_PERM = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]

SMPLX_JOINTS_FLIP_PERM = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13,
                        15, 17, 16, 19, 18, 21, 20, 22, 24, 23,
                        40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
SMPLX_POSE_FLIP_PERM = []
for i in SMPLX_JOINTS_FLIP_PERM:
    SMPLX_POSE_FLIP_PERM.append(3*i)
    SMPLX_POSE_FLIP_PERM.append(3*i+1)
    SMPLX_POSE_FLIP_PERM.append(3*i+2)

def flip_pose(pose):
    #Flip pose.The flipping is based on SMPLX parameters.
    pose = pose[:,SMPLX_POSE_FLIP_PERM]
    # we also negate the second and the third dimension of the axis-angle
    pose[:,1::3] = -pose[:,1::3]
    pose[:,2::3] = -pose[:,2::3]
    return pose

def get_train_test_list(datasplit):
        all_list = []
        train_list = []
        for i in range(1,212):
            all_list.append(str(i).zfill(3))
        
        if datasplit == "cross_genre":
            test_list = ["063", "132", "143", "036", "098", "198", "130", "012", "211", "193", "179", "065", "137", "161", "092", "037", "109", "204", "144"]
            ignor_list = ["116", "117", "118", "119", "120", "121", "122", "123", "202"]+["130"]
        elif datasplit == "cross_dancer":
            test_list = ['001','002','003','004','005','006','007','008','009','010','011','012','013','124','126','128','130','132']
            ignor_list = ['115','117','119','121','122','135','137','139','141','143','145','147'] + ["116", "118", "120", "123", "202", "159"]+["130"]       # 前一个列表为val set，后一个列表为ignore set
        else:
            raise("error of data split!")
        for one in all_list:
            if one not in test_list:
                if one not in ignor_list:
                    train_list.append(one)
        return train_list, test_list, ignor_list

def tem_data():
        train_list = ['014','015','016','017']
        test_list = ['063']
        ignor_list =[]

        return train_list, test_list, ignor_list

    



class FineDance_Smpl(data.Dataset):
    def __init__(self, args, istrain):
        self.motion_dir =args.motion_dir
        self.music_fm_dir = args.music_fm_dir
        self.music_basic_dir = args.music_basic_dir
        self.istrain = istrain
        self.seq_len = args.full_seq_len
        slide = args.full_seq_len // args.windows
        json_file = args.genre_json
        total_length = 0            # 将数据集中的所有motion用同一个index索引
        self.motion_index = []
        self.music_index = []
        self.text_all  = []

        self.name = []
        self.genre = {}
        motion_all = []
        music_all = []
        
        #train_list, test_list, ignor_list = get_train_test_list(args.datasplit)
        train_list, test_list, ignor_list = tem_data()
        
        if self.istrain:
            self.datalist= train_list
        else:
            self.datalist = test_list

        with open(json_file, 'r') as file:
            styles = json.load(file)
        self.name_to_style = {music['filename']: music['style2'] for music in styles}

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            
        for name in tqdm(self.datalist):
            save_name = name
            name = name + ".npy"
            if name[:-4] in ignor_list:
                continue
            motion = np.load(os.path.join(self.motion_dir, name))
            fm_music = np.load(os.path.join(self.music_fm_dir, name))
            music_basic = np.load(os.path.join(self.music_basic_dir, name)).T

            if len(music_basic) < len(fm_music):
                fm_music = fm_music[:len(music_basic)]

            min_all_len = min(motion.shape[0], fm_music.shape[0])
            motion = motion[:min_all_len]
            if motion.shape[-1] == 168:
                motion = np.concatenate([motion[:,:69], motion[:,78:]], axis=1)     # 22,  25
            elif motion.shape[-1] == 319:
                pass
            elif motion.shape[-1] == 315:
                pass
            else:
                raise("input motion shape error! not 168 or 319!")
            fm_music = fm_music[:min_all_len]        
            music_basic = music_basic[:min_all_len]  
            music = np.concatenate((fm_music, music_basic), axis=1)
            nums = (min_all_len-self.seq_len) // slide + 1  
            if self.istrain:
                clip_index = []
                for i in range(nums):
                    motion_clip = motion[i * slide: i * slide + self.seq_len]
                    if motion_clip.std(axis=0).mean() > 0.07:          
                        clip_index.append(i)
                index = np.array(clip_index) * slide + total_length    
                index_ = np.array(clip_index) * slide
            else:
                index = np.arange(nums) * slide + total_length
                index_ = np.arange(nums) * slide 

            motion_all.append(motion)
            music_all.append(music)
            if args.mix:
                motion_index = []
                music_index = []
                num = (len(index) - 1) // 8 + 1
                for i in range(num):
                    motion_index_tmp, music_index_tmp = np.meshgrid(index[i*8:(i+1)*8], index[i*8:(i+1)*8])         
                    motion_index += motion_index_tmp.reshape((-1)).tolist()
                    music_index += music_index_tmp.reshape((-1)).tolist()
                    index_tmp = np.meshgrid(index_[i*8:(i+1)*8])
                    index_ += index_tmp.reshape((-1)).tolist()

            else:
                motion_index = index.tolist()
                music_index = index.tolist()
                index_ = index_.tolist()
            

            genre = self.name_to_style[save_name]
            prompt1 = 'This is a '
            prompt2 = ' type of music.'
            y = [prompt1 + genre + prompt2]
            inputs = tokenizer(y, padding=True, return_tensors="pt")
            text_embeddings = model.get_text_features(**inputs)
            
            for i in index_:
                self.text_all.append(text_embeddings.detach().numpy())
            

            index_ = [save_name + "_" + str(element).zfill(5) for element in index_]
            self.motion_index += motion_index
            self.music_index += music_index
            total_length += min_all_len
            self.name += index_

        self.text_all =  np.concatenate(self.text_all, axis=0).astype(np.float32)
        self.motion = np.concatenate(motion_all, axis=0).astype(np.float32)
        self.music = np.concatenate(music_all, axis=0).astype(np.float32)
        self.len = len(self.motion_index)
        print(f'FineDance has {self.len} samples..')

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        motion_index = self.motion_index[index]
        music_index = self.music_index[index]
        motion = self.motion[motion_index:motion_index+self.seq_len]
        if motion.shape[-1] == 319 or motion.shape[-1] == 139:
            motion[:, 4:7]  = motion[:, 4:7] - motion[:1, 4:7]           # The first 4 dimension are foot contact
        else:
            motion[:, :3] = motion[:, :3] - motion[:1, :3]
        music = self.music[music_index:music_index+self.seq_len]
        filename = self.name[index]
        text = self.text_all[index]
        return motion, music, filename, text

    

if __name__ == '__main__':
    data_split = {}
    all_list = []
    train_list = []
    for i in range(1,212):
        all_list.append(str(i).zfill(3))
    test_list = ["063", "132", "143", "036", "098", "198", "130", "012", "211", "193", "179", "065", "137", "161", "092", "120", "037", "109", "204", "144"]
    val_list = ["116", "117", "118", "119", "120", "121", "122", "123", "202"]
    for one in all_list:
        if one not in test_list:
            if one not in val_list:
                train_list.append(one)

    data_split["train"] = train_list
    data_split["test"] = test_list
    data_split["val"] = val_list
    data_split["ignore"] =  ["130"]

    with open("data_crossgenre.json", "w") as f:
        json.dump(data_split,f) 


    print(train_list)