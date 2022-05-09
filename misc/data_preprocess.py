from pathlib import Path
import shutil
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
cwd = os.getcwd()


# number of samples per class to select
N = 300

# train (val test) split
SPLIT = 1/2

# size for cutting out of frames
CUT_SIZE = 400

# size of frames to be fed to the network
SIZE = 256


p = Path('/media/hdd2/matthias')
p_action_label = p / 'datasets' / 'interactions'
p_bb = p / 'monkey_vids_output'
p_vids = p / 'monkey_vids'


    


def get_data(p):
    pose_df = get_action_df(p)
    bb_df = get_bb_df(p)

    joined_df = get_joined_df(pose_df, bb_df)

    labels = joined_df.pose.unique()

    get_pose_info(joined_df, labels)
    samples = get_samples(joined_df, labels)
    # get_sample_info(samples, labels)
    samples_split = get_split(samples, labels)

    return samples_split, labels



def get_bb_df(p):
    pos = p.name.rfind('_')
    bb_df = pd.read_csv(str((p_bb / p.stem[:3] / (p.name[:pos]+'.mp4') / 'results.txt')), sep=',', names=['frame','id','x1','y1','w','h','score','cls_id','1'])

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

    return bb_df



def get_action_df(p):
    label_df = pd.read_csv(str(p), sep=' ', names=['frame','from','to','pose_or_action'])

    # drop all rows containing interactions
    label_df = label_df[label_df.to == '-']

    # drop to column as it is no longer needed
    label_df.drop('to', axis=1, inplace=True)

    # rename last column as contents are now cleaned up
    label_df.rename(columns={'pose_or_action': 'pose'}, inplace=True)

    # can drop from column there is only one monkey either way...even if it is detected with multiple ID's
    label_df.drop('from', axis=1, inplace=True)

    # set frame column as index column
    label_df.set_index('frame', inplace=True)

    return label_df



def get_joined_df(label_df, bb_df):
    joined_df = bb_df.merge(label_df, left_index=True, right_index=True)

    # turn bb values to int
    joined_df[['x1','y1','w','h']] = joined_df[['x1','y1','w','h']].astype(int)

    return joined_df


def get_pose_info(joined_df, labels):
    for l in labels:
        print(l, len(joined_df[joined_df.pose == l]))


def get_samples(joined_df, labels):
    # sample N elements per pose
    samples = {l: joined_df[joined_df.pose==l].sample(N).drop('pose', axis=1)
                            for l in labels
                        }

    return samples

def get_sample_info(samples, labels):
    print(samples[labels[0]].iloc[0])

def get_split(samples, labels): 
    # # create train and val split
    samples_train = {l: samples[l].iloc[:int(SPLIT*N)]
                            for l in labels
                        }
    samples_validation = {l: samples[l].iloc[int(SPLIT*N):int((1-SPLIT*SPLIT)*N)]
                            for l in labels
                        }
    samples_test = {l: samples[l].iloc[int((1-SPLIT*SPLIT)*N):]
                            for l in labels
                        }

    samples = {'train': samples_train, 'validation': samples_validation, 'test': samples_test}

    return samples




def get_frames(p, samples, labels):
    pos_sep = p.name.rfind('_')
    cap = cv2.VideoCapture(str((p_vids / p.stem[:3] / p.name[:pos_sep] / p.stem[pos_sep+1:])))

    for set in ['train', 'validation', 'test']:
        for l in labels:
            for index, bb_info in samples[set][l].iterrows():

                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ret, frame = cap.read()

                cut = cut_frame(frame, bb_info)

                padded = pad_frame(cut)

                cv2.imwrite(str((p.parents[1] / 'monkey_poses' / set / l))+"/"+p.name[:-8]+"_{:05d}.png".format(index), padded)


def cut_frame(frame, bb_info):
    x1, y1, w, h = bb_info
    (fh, fw, c) = frame.shape
            
    y = np.clip([y1 - (CUT_SIZE-h)//2, y1 + h + (CUT_SIZE-h)//2], 0, fh)
    x = np.clip([x1 - (CUT_SIZE-w)//2, x1 + w + (CUT_SIZE-w)//2], 0, fw)
    frame = frame[y[0]:y[1],x[0]:x[1],:]

    return frame


def pad_frame(frame):
    w_pad, h_pad = max(0, CUT_SIZE - frame.shape[0]), max(0, CUT_SIZE - frame.shape[1])

    if w_pad > 0 or h_pad > 0:
        frame = np.pad(frame, ((w_pad//2, w_pad//2+w_pad%2), (h_pad//2, h_pad//2+h_pad%2), (0,0)), 'constant')
    
    return frame   



def cleanup():
    p = Path('/media/hdd2/matthias/datasets/monkey_poses/')
    path_list = p.glob("**/*")
    [f.unlink() for f in path_list if f.is_file()]


cleanup()

for p in p_action_label.glob('*.txt'):

    print(str(p))
    samples, labels = get_data(p)

    get_frames(p, samples, labels)