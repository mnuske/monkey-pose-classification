from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import cv2
import logging
import argparse
import numpy as np
import torch

from inference_utils import visualization as vis
from inference_utils.log import logger
from inference_utils.timer import Timer
import datasets.jde as datasets

from inference_utils.utils import mkdir_if_missing
from opts import opts


def write_results(filename, results, data_type):
    if data_type == 'mpc':
        save_format = '{frame},{pose},{x1},{y1},{w},{h}\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, (tlwh, pose) in results:
            if tlwh is None and pose is None:
                line = save_format.format(frame=frame_id, pose=' ', x1=' ', y1=' ', w=' ', h=' ')
            else:
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, pose=pose, x1=x1, y1=y1, w=w, h=h)
            f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results_score(filename, results, data_type):
    if data_type == 'mpc':
        save_format = '{frame},{pose},{x1},{y1},{w},{h},{s}\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, (tlwh, pose, score) in results:
            if tlwh is None and pose is None:
                line = save_format.format(frame=frame_id, pose=' ', x1=' ', y1=' ', w=' ', h=' ', s=' ')
            else:
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, pose=pose, x1=x1, y1=y1, w=w, h=h, s=score)
            f.write(line)
    logger.info('save results to {}'.format(filename))

# @torch.no_grad()
def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=70, use_cuda=True, device='gpu'):
    if save_dir:
        mkdir_if_missing(save_dir)
    # res_model = torch.load(opt.load_model)
    res_model = torch.load('/home/matthias/monkey/monkey-pose-classification/model_own/model_final.pth')
    res_model.eval()
    res_model.to(device)
    # if use_cuda:
    #     res_model = res_model.cuda
    timer = Timer()
    results = []
    frame_id = 0
    #for path, img, img0 in dataloader:
    for i, data in enumerate(dataloader):# dataloader should return (image and bb cutout) or none
        if frame_id % 4200 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        
        (img, tlwh, img0) = data
        if img is None:
            break
        if tlwh is None and img0 is None:
            results.append((frame_id, (None, None)))
            online_im = img
        else:

            # run classification
            timer.tic()
            if use_cuda:
                # input = torch.from_numpy(img0).cuda
                input = img0.to(device)
            else:
                # input = torch.from_numpy(img0)
                input = img0
            
            with torch.no_grad():
                output = res_model(input.unsqueeze(0)).to('cpu')
                # output = res_model(input.unsqueeze(0))
                # output = torch.sigmoid(output)
                # output = (output/output.sum()).to('cpu')

            _, pred = torch.max(output, 1)

            timer.toc()
            # save results
            results.append((frame_id, (tlwh, pred)))
            #results.append((frame_id, (tlwh, pred, output)))

            if show_image or save_dir is not None:
                online_im = vis.plot_tracking(opt, img, tlwh, pred, output, frame_id=frame_id,
                                            fps=1. / timer.average_time)
        
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
        # save results
    write_results(result_filename, results, data_type)
    #write_results_score(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls
