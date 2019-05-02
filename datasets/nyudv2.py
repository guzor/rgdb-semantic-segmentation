from torch.utils.data import Dataset
# import glob
import numpy as np
import cv2
import h5py

N_IMAGES = 10


class Dataset(Dataset):
    def __init__(self, flip_prob=None, crop_type=None, crop_size=0):

        self.flip_prob = flip_prob
        self.crop_type = crop_type
        self.crop_size = crop_size

        data_path = 'datasets/data/'
        data_file = 'nyu_depth_v2_labeled.mat'

        # read mat file
        print("Reading .mat file...")

        # as mat
        # import scipy.io
        # f = scipy.io.loadmat(data_path + data_file)
        # rgb_images_fr = np.transpose(f['images'][:, :, :, 0:N_IMAGES], [3, 1, 0, 2]).astype(np.float32)
        # label_images_fr = np.array(f['labels'][:, :, 0:N_IMAGES])
        # self.label_names = np.array(['<UNK>'] + [x[0][0] for x in f['names']])
        # print(self.label_names)

        # as h5
        f = h5py.File(data_path + data_file)

        rgb_images_fr = np.transpose(f['images'][0:N_IMAGES], [0, 2, 3, 1]).astype(np.float32)
        label_images_fr = np.array(f['labels'][0:N_IMAGES])
        self.label_names = np.array(['<UNK>'] + self.get_names(f))
        f.close()

        self.rgb_images = rgb_images_fr
        self.label_images = label_images_fr

    def get_names(self, h5py_data, field='names'):
        extract_name = lambda index: ''.join([chr(v) for v in h5py_data[h5py_data[field][0][index]]])
        return [extract_name(i) for i in range(len(h5py_data[field][0]))]

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        rgb = self.rgb_images[idx].astype(np.float32)
        index = 5000 + idx + 1
        hha = np.transpose(cv2.imread("datasets/data/hha/img_" + str(index) + ".png", cv2.COLOR_BGR2RGB), [1, 0, 2])
        # TODO sizes aren't the same- cropped to (46:470, 41:600, :) in Depth2HHA/utils/nyu-hooks/cropIt.m
        rgb_hha = np.concatenate([rgb, hha], axis=2).astype(np.float32)
        label = self.label_images[idx].astype(np.float32)
        label[label >= len(self.label_names)] = 0
        xy = np.zeros_like(rgb)[:, :, 0:2].astype(np.float32)

        # random crop
        if self.crop_type is not None and self.crop_size > 0:
            max_margin = rgb_hha.shape[0] - self.crop_size
            if max_margin == 0:  # crop is original size, so nothing to crop
                self.crop_type = None
            elif self.crop_type == 'Center':
                rgb_hha = rgb[max_margin // 2:-max_margin // 2, max_margin // 2:-max_margin // 2, :]
                label = label[max_margin // 2:-max_margin // 2, max_margin // 2:-max_margin // 2]
                xy = xy[max_margin // 2:-max_margin // 2, max_margin // 2:-max_margin // 2, :]
            elif self.crop_type == 'Random':
                x_ = np.random.randint(0, max_margin)
                y_ = np.random.randint(0, max_margin)
                rgb_hha = rgb_hha[y_:y_ + self.crop_size, x_:x_ + self.crop_size, :]
                label = label[y_:y_ + self.crop_size, x_:x_ + self.crop_size]
                xy = xy[y_:y_ + self.crop_size, x_:x_ + self.crop_size, :]
            else:
                print('Bad crop')  # TODO make this more like, you know, good software
                exit(0)

        # random flip
        if self.flip_prob is not None:
            if np.random.random() > self.flip_prob:
                rgb_hha = np.fliplr(rgb_hha).copy()
                label = np.fliplr(label).copy()
                xy = np.fliplr(xy).copy()

        return rgb_hha, label, xy
