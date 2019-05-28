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
from datasets import nyud2headed
from models import Model
from model2headed import Model2Headed
import config
from tqdm import tqdm
import argparse

torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('3dgnn')
    parser.add_argument('--num_epochs', default=50, type=int,
                        help='Number of epoch')
    parser.add_argument('--batchsize', type=int, default=4,
                        help='batch size in training')
    parser.add_argument('--pretrain', type=str, default=None,
                        help='Direction for pretrained weight')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')
    parser.add_argument('--is_2_headed', type=bool, default=False,
                        help='which model to use')

    return parser.parse_args()


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    logger = logging.getLogger('3dgnn')
    log_path = './experiment/' + str(datetime.datetime.now().strftime('%Y-%m-%d-%H')).replace(' ', '/') + '/'
    print('log path is:', log_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        os.makedirs(log_path + 'save/')
    hdlr = logging.FileHandler(log_path + 'log.txt')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    logger.info("Loading data...")
    print("Loading data...")

    '''idx_to_label = {0: '<UNK>', 1: 'beam', 2: 'board', 3: 'bookcase', 4: 'ceiling', 5: 'chair', 6: 'clutter',
                    7: 'column',
                    8: 'door', 9: 'floor', 10: 'sofa', 11: 'table', 12: 'wall', 13: 'window'}*'''

    if args.is_2_headed:
        dataset_tr = nyud2headed.Dataset(flip_prob=config.flip_prob, crop_type='Random', crop_size=config.crop_size)
    else:
        dataset_tr = nyudv2.Dataset(flip_prob=config.flip_prob, crop_type='Random', crop_size=config.crop_size)
    idx_to_label = dataset_tr.label_names
    if args.is_2_headed:
        idx_to_label2 = dataset_tr.label2_names

    dataloader_tr = DataLoader(dataset_tr, batch_size=args.batchsize, shuffle=True,
                               num_workers=config.workers_tr, drop_last=False, pin_memory=True)

    if args.is_2_headed:
        dataset_va = nyud2headed.Dataset(flip_prob=0.0, crop_type='Center', crop_size=config.crop_size)
    else:
        dataset_va = nyudv2.Dataset(flip_prob=0.0, crop_type='Center', crop_size=config.crop_size)
    dataloader_va = DataLoader(dataset_va, batch_size=args.batchsize, shuffle=False,
                               num_workers=config.workers_va, drop_last=False, pin_memory=True)
    cv2.setNumThreads(config.workers_tr)

    logger.info("Preparing model...")
    print("Preparing model...")

    class_weights = [0.0] + [1.0 for i in range(1, len(idx_to_label))]
    nclasses = len(class_weights)
    if args.is_2_headed:
        nclasses1 = nclasses
        class2_weights = [0.0] + [1.0 for i in range(1, len(idx_to_label2))]
        nclasses2 = len(class2_weights)
        model = Model2Headed(nclasses1, nclasses2, config.mlp_num_layers, config.use_gpu)
        loss2 = nn.NLLLoss(reduce=not config.use_bootstrap_loss, weight=torch.FloatTensor(class2_weights))
    else:
        model = Model(nclasses, config.mlp_num_layers, config.use_gpu)
    loss = nn.NLLLoss(reduce=not config.use_bootstrap_loss, weight=torch.FloatTensor(class_weights))

    softmax = nn.Softmax(dim=1)
    log_softmax = nn.LogSoftmax(dim=1)

    if config.use_gpu:
        model = model.cuda()
        loss = loss.cuda()
        if args.is_2_headed:
            loss2 = loss2.cuda()
        softmax = softmax.cuda()
        log_softmax = log_softmax.cuda()

    optimizer = torch.optim.Adam([{'params': model.decoder.parameters()},
                                  {'params': model.gnn.parameters(), 'lr': config.gnn_initial_lr}],
                                 lr=config.base_initial_lr, betas=config.betas, eps=config.eps,
                                 weight_decay=config.weight_decay)

    if config.lr_schedule_type == 'exp':
        lambda1 = lambda epoch: pow((1 - ((epoch - 1) / args.num_epochs)), config.lr_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    elif config.lr_schedule_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.lr_decay,
                                                               patience=config.lr_patience)
    else:
        print('bad scheduler')
        exit(1)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info("Number of trainable parameters: %d", params)

    def get_current_learning_rates():
        learning_rates = []
        for param_group in optimizer.param_groups:
            learning_rates.append(param_group['lr'])
        return learning_rates

    def eval_set(dataloader):
        model.eval()

        with torch.no_grad():
            loss_sum = 0.0
            if config.use_gpu:
                confusion_matrix = torch.cuda.FloatTensor(np.zeros(len(idx_to_label) ** 2))
            else:
                confusion_matrix = torch.FloatTensor(np.zeros(len(idx_to_label) ** 2))

            start_time = time.time()

            for batch_idx, rgbd_label_xy in tqdm(enumerate(dataloader), total=len(dataloader), smoothing=0.9):
                x = rgbd_label_xy[0]
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
                # if args.is_2_headed:
                #     output1, output2 = model(input, gnn_iterations=config.gnn_iterations, k=config.gnn_k, xy=xy,
                #                              use_gnn=config.use_gnn)

                if config.use_bootstrap_loss:
                    loss_per_pixel = loss.forward(log_softmax(output.float()), target)
                    topk, indices = torch.topk(loss_per_pixel.view(output.size()[0], -1),
                                               int((config.crop_size ** 2) * config.bootstrap_rate))
                    loss_ = torch.mean(topk)
                else:
                    loss_ = loss.forward(log_softmax(output.float()), target)
                loss_sum += loss_

                pred = output.permute(0, 2, 3, 1).contiguous()
                pred = pred.view(-1, nclasses)
                pred = softmax(pred)
                pred_max_val, pred_arg_max = pred.max(1)

                pairs = target.view(-1) * len(idx_to_label) + pred_arg_max.view(-1)
                for i in range(len(idx_to_label) ** 2):
                    cumu = pairs.eq(i).float().sum()
                    confusion_matrix[i] += cumu.item()

            sys.stdout.write(" - Eval time: {:.2f}s \n".format(time.time() - start_time))
            loss_sum /= len(dataloader)

            confusion_matrix = confusion_matrix.cpu().numpy().reshape((len(idx_to_label), len(idx_to_label)))
            class_iou = np.zeros(len(idx_to_label))
            confusion_matrix[0, :] = np.zeros(len(idx_to_label))
            confusion_matrix[:, 0] = np.zeros(len(idx_to_label))
            for i in range(1, len(idx_to_label)):
                tot = np.sum(confusion_matrix[i, :]) + np.sum(confusion_matrix[:, i]) - confusion_matrix[i, i]
                if tot == 0:
                    class_iou[i] = 0
                else:
                    class_iou[i] = confusion_matrix[i, i] / tot

        return loss_sum.item(), class_iou, confusion_matrix

    '''Training parameter'''
    model_to_load = args.pretrain
    logger.info("num_epochs: %d", args.num_epochs)
    print("Number of epochs: %d" % args.num_epochs)
    interval_to_show = 100

    train_losses = []
    eval_losses = []

    if model_to_load:
        logger.info("Loading old model...")
        print("Loading old model...")
        model.load_state_dict(torch.load(model_to_load))
    else:
        # print("here")
        # exit(0)
        logger.info("Starting training from scratch...")
        print("Starting training from scratch...")

    '''Training'''
    for epoch in range(1, args.num_epochs + 1):
        print("epoch", epoch)
        batch_loss_avg = 0
        if config.lr_schedule_type == 'exp':
            scheduler.step(epoch)
        for batch_idx, rgbd_label_xy in tqdm(enumerate(dataloader_tr), total=len(dataloader_tr), smoothing=0.9):
            x = rgbd_label_xy[0]
            target = rgbd_label_xy[1].long()
            if args.is_2_headed:
                target2 = rgbd_label_xy[3].long()
            xy = rgbd_label_xy[2]
            x = x.float()
            xy = xy.float()

            input = x.permute(0, 3, 1, 2).contiguous()
            input = input.type(torch.FloatTensor)

            if config.use_gpu:
                input = input.cuda()
                xy = xy.cuda()
                target = target.cuda()
                if args.is_2_headed:
                    target2 = target2.cuda()

            xy = xy.permute(0, 3, 1, 2).contiguous()

            optimizer.zero_grad()
            model.train()

            if args.is_2_headed:
                output1, output2 = model(input, gnn_iterations=config.gnn_iterations, k=config.gnn_k, xy=xy,
                                         use_gnn=config.use_gnn)
            else:
                output = model(input, gnn_iterations=config.gnn_iterations, k=config.gnn_k, xy=xy,
                               use_gnn=config.use_gnn)

            if config.use_bootstrap_loss:
                loss_per_pixel = loss.forward(log_softmax(output.float()), target)
                topk, indices = torch.topk(loss_per_pixel.view(output.size()[0], -1),
                                           int((config.crop_size ** 2) * config.bootstrap_rate))
                loss_ = torch.mean(topk)
            else:
                if args.is_2_headed:
                    loss_ = loss.forward(log_softmax(output1.float()), target) + loss2.forward(
                        log_softmax(output2.float()), target2)
                else:
                    loss_ = loss.forward(log_softmax(output.float()), target)

            loss_.backward()
            optimizer.step()

            batch_loss_avg += loss_.item()

            if batch_idx % interval_to_show == 0 and batch_idx > 0:
                batch_loss_avg /= interval_to_show
                train_losses.append(batch_loss_avg)
                logger.info("E%dB%d Batch loss average: %s", epoch, batch_idx, batch_loss_avg)
                print('\rEpoch:{}, Batch:{}, loss average:{}'.format(epoch, batch_idx, batch_loss_avg))
                batch_loss_avg = 0

        batch_idx = len(dataloader_tr)
        logger.info("E%dB%d Saving model...", epoch, batch_idx)

        torch.save(model.state_dict(), log_path + '/save/' + 'checkpoint_' + str(epoch) + '.pth')

        '''Evaluation'''
        # eval_loss, class_iou, confusion_matrix = eval_set(dataloader_va)
        # eval_losses.append(eval_loss)
        #
        # if config.lr_schedule_type == 'plateau':
        #     scheduler.step(eval_loss)
        print('Learning ...')
        logger.info("E%dB%d Def learning rate: %s", epoch, batch_idx, get_current_learning_rates()[0])
        print('Epoch{} Def learning rate: {}'.format(epoch, get_current_learning_rates()[0]))
        logger.info("E%dB%d GNN learning rate: %s", epoch, batch_idx, get_current_learning_rates()[1])
        print('Epoch{} GNN learning rate: {}'.format(epoch, get_current_learning_rates()[1]))
        # logger.info("E%dB%d Eval loss: %s", epoch, batch_idx, eval_loss)
        # print('Epoch{} Eval loss: {}'.format(epoch, eval_loss))
        # logger.info("E%dB%d Class IoU:", epoch, batch_idx)
        # print('Epoch{} Class IoU:'.format(epoch))
        # for cl in range(len(idx_to_label)):
        #     logger.info("%+10s: %-10s" % (idx_to_label[cl], class_iou[cl]))
        #     print('{}:{}'.format(idx_to_label[cl], class_iou[cl]))
        # logger.info("Mean IoU: %s", np.mean(class_iou[1:]))
        # print("Mean IoU: %.2f" % np.mean(class_iou[1:]))
        # logger.info("E%dB%d Confusion matrix:", epoch, batch_idx)
        # logger.info(confusion_matrix)

    logger.info("Finished training!")
    logger.info("Saving model...")
    print('Saving final model...')
    torch.save(model.state_dict(), log_path + '/save/3dgnn_finish.pth')
    # eval_loss, class_iou, confusion_matrix = eval_set(dataloader_va)
    # logger.info("Eval loss: %s", eval_loss)
    # logger.info("Class IoU:")
    # for cl in range(len(idx_to_label)):
    #     logger.info("%+10s: %-10s" % (idx_to_label[cl], class_iou[cl]))
    # logger.info("Mean IoU: %s", np.mean(class_iou[1:]))


if __name__ == '__main__':
    args = parse_args()
    main(args)
