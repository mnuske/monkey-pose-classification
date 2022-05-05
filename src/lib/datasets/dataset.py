from pathlib import Path
import shutil
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
cwd = os.getcwd()

import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset
import torchvision

from PIL import Image


# size for cutting out of frames
CUT_SIZE = 400

# size of frames to be fed to the network
SIZE = 256


p = Path('/media/hdd2/matthias')

p_action_label = p / 'datasets' / 'interactions'

p_bb = p / 'monkey_vids_output'
p_vids = p / 'monkey_vids'


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


transform = {
    'train': transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
                    ]),
    'validation': transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize,
                        ]),
    'test': transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize,
    ])
}


class LoadVideo(Dataset):

    def __init__(self, opt):
        self.video_path = opt.video_path 
        self.bb_path = opt.bb_path

        self.video = cv2.VideoCapture(self.video_path)
        self.frame_rate = self.video.get(cv2.CAP_PROP_FPS)
        
        self.bbs = self.get_bb_df(self.bb_path)
        self.index = 0

        # self.video.set(cv2.CAP_PROP_POS_FRAMES, 19200)
        # self.logger = logger

    
    def __len__(self):
        return int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))-1


    def __getitem__(self, idx):

        ret, frame = self.video.read()

        try:
            xywh = self.bbs.loc[self.index]
        except KeyError as e:
            return frame, None, None

        cut = self.cut_frame(frame, xywh)

        padded = self.pad_frame(cut)

        pil = Image.fromarray(padded)
        
        if transform:
            img0 = transform['test'](pil)

        self.index += 1

        return frame, xywh, img0



    @staticmethod
    def get_bb_df(bb_path):
        bb_df = pd.read_csv(bb_path, sep=',', names=['frame','id','x1','y1','w','h','score','cls_id','1'])

        # drop last column thats always 1 and score as we ingnore it either way
        bb_df.drop(['1', 'score'], axis=1, inplace=True)

        # drop all rows not containing a bb for a monkey
        bb_df = bb_df[bb_df.cls_id == 0]

        # drop column containing cls_id
        bb_df.drop('cls_id', axis=1, inplace=True)

        # substract 1 from frame column
        bb_df.frame -= 1

        # drop id column as we do not have a need for it
        bb_df.drop('id', axis=1, inplace=True)

        # set frame column as index column
        bb_df.set_index('frame', inplace=True)

        # get fields to type int
        bb_df[['x1','y1','w','h']] = bb_df[['x1','y1','w','h']].astype(int)

        return bb_df


    @staticmethod
    def cut_frame(frame, bb_info):
        x1, y1, w, h = bb_info
        (fh, fw, c) = frame.shape
                
        y = np.clip([y1 - (CUT_SIZE-h)//2, y1 + h + (CUT_SIZE-h)//2], 0, fh)
        x = np.clip([x1 - (CUT_SIZE-w)//2, x1 + w + (CUT_SIZE-w)//2], 0, fw)
        frame = frame[y[0]:y[1],x[0]:x[1],:]

        return frame


    @staticmethod
    def pad_frame(frame):
        w_pad, h_pad = max(0, CUT_SIZE - frame.shape[0]), max(0, CUT_SIZE - frame.shape[1])

        if w_pad > 0 or h_pad > 0:
            frame = np.pad(frame, ((w_pad//2, w_pad//2+w_pad%2), (h_pad//2, h_pad//2+h_pad%2), (0,0)), 'constant')
        
        return frame   