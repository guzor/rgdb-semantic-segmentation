import cv2
import os
import sys
import time
import numpy as np
import datetime
import logging
import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import nyudv2
from models import Model
import config
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt


def main(args):
    dataset = nyudv2.Dataset()
    idx_to_label = dataset.label_names
    dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False,
                            num_workers=config.workers_tr, drop_last=False, pin_memory=True)

    class_weights = [0.0] + [1.0 for i in range(1, len(idx_to_label))]
    nclasses = len(class_weights)
    model = Model(nclasses, config.mlp_num_layers, config.use_gpu)
    print("Loading model...")
    model.load_state_dict(torch.load(args.path, map_location=lambda storage, loc: storage))
    print("Finsihed Loading model...")

    softmax = nn.Softmax(dim=1)
    confusion_matrix = torch.FloatTensor(np.zeros(len(idx_to_label) ** 2))
    segmented_path = './segmented/' + str(datetime.datetime.now().strftime('%Y-%m-%d-%H')).replace(' ', '/') + '/'
    if not os.path.exists(segmented_path):
        os.makedirs(segmented_path)

    for batch_idx, rgbd_label_xy in tqdm(enumerate(dataloader), total=len(dataloader), smoothing=0.9):
        x = rgbd_label_xy[0]  # rgb_hha, label, xy
        xy = rgbd_label_xy[2]
        target = rgbd_label_xy[1].long()
        x = x.float()
        xy = xy.float()

        input = x.permute(0, 3, 1, 2).contiguous()
        xy = xy.permute(0, 3, 1, 2).contiguous()
        if config.use_gpu:
            input = input.cuda()
            xy = xy.cuda()
            target = target.cuda()

        output = model(input, gnn_iterations=config.gnn_iterations, k=config.gnn_k, xy=xy,
                       use_gnn=config.use_gnn)
        pred = output.permute(0, 2, 3, 1).contiguous()
        pred = pred.view(-1, nclasses)
        pred = softmax(pred)
        pred_max_val, pred_arg_max = pred.max(1)

        for i in range(1, len(idx_to_label)):
            current_show = pred_arg_max.view(4, 640, 480).permute(0, 2, 1)
            nt = current_show.numpy()
            origin_cur = input.permute(0, 3, 2, 1)[:, :, :, 0:3].numpy().astype(np.int)
            mask = np.equal(i, nt)
            res = origin_cur
            res[:, :, :, 1] = res[:, :, :, 1] + 0.4 * 255 * mask
            for j in range(args.batchsize):
                plt.imsave(segmented_path + str(5001 + batch_idx * args.batchsize + j) + '_' + idx_to_label[i],
                           res[j].astype(np.uint8))


def parse_args():
    parser = argparse.ArgumentParser('3dgnn')
    parser.add_argument('--path', default="trained_models/3dgnn_finish.pth", type=str,
                        help='Model\'s file path')
    parser.add_argument('--batchsize', type=int, default=4,
                        help='batch size in training')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
