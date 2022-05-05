from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import logging
import os
import os.path as osp
from opts import opts
from inference_utils.utils import mkdir_if_missing
from inference_utils.log import logger
import datasets.dataset as dataset
from inference import eval_seq
import torch


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logger.setLevel(logging.INFO)


def demo(opt):
    cuda0 = torch.device('cuda:0')

    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    logger.info('Starting inference...')
    dataloader = dataset.LoadVideo(opt)
    result_filename = os.path.join(result_root, 'pose.txt')
    frame_rate = dataloader.frame_rate

    frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')
    # TODO what does show_image do?
    eval_seq(opt, dataloader, 'mpc', result_filename,
             save_dir=frame_dir, show_image=False, frame_rate=frame_rate,
             use_cuda=opt.gpus!=[-1], device=cuda0)

    if opt.output_format == 'video':
        output_video_path = osp.join(result_root, opt.output_name + '.mp4')
        cmd_str = 'ffmpeg -y -f image2 -i {}/%05d.jpg -b:v 5000k -c:v mpeg4 -vf fps={} {}'.format(osp.join(result_root, 'frame'), dataloader.frame_rate, output_video_path)
        os.system(cmd_str)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt = opts().init()
    demo(opt)