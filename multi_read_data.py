import numpy as np
import torch
import torch.utils.data
import random
from PIL import Image
from glob import glob
import torchvision.transforms as transforms
import os
import cv2
from torchvision.transforms import Compose, ToTensor
import torch.nn.functional as F
import random

batch_w = 600
batch_h = 400

def transform():
    return Compose([
        ToTensor(),
        #Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
from os import listdir
from os.path import join

class MemoryFriendlyLoader_train(torch.utils.data.Dataset):
    def __init__(self, img_dir, task):
        self.low_img_dir = img_dir
        self.task = task
        self.train_low_data_names = []
        dir_high = self.low_img_dir + "high"
        dir_low = self.low_img_dir + "low"
        self.hr_image_filenames = [join(dir_high, x) for x in listdir(dir_high)]
        self.lr_image_filenames = [join(dir_low, x) for x in listdir(dir_high)]

        dir_sti = self.low_img_dir + "Sti"
        self.sti_image_filenames = [join(dir_sti, x) for x in listdir(dir_sti)]

        self.count = len(self.lr_image_filenames)# + len(self.sti_image_filenames)

        transform_list = []
        transform_list += [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)
        self.trans = transforms.ToTensor()

    def load_images_transform(self, file):
        im = Image.open(file).convert("HSV")
        img_norm = self.transform(im).numpy()
        img_norm = np.transpose(img_norm, (1, 2, 0))
        # im_hsv = cv2.cvtColor(img_norm, cv2.COLOR_RGB2HSV)
        return img_norm

    def __getitem__(self, index):
        if index < len(self.lr_image_filenames):
            low_i = cv2.imread(self.lr_image_filenames[index])  # .convert('RGB')
            low_1 = low_i[:, :, ::-1].copy()
            low = self.trans(low_1)
            high_in = cv2.imread(self.hr_image_filenames[index])  # .convert('RGB')
            high_1 = high_in[:, :, ::-1].copy()
            high = self.trans(high_1)
            img_name = self.lr_image_filenames[index].split('\\')[-1]
        else:
            high_in = cv2.imread(self.sti_image_filenames[index-len(self.lr_image_filenames)])  # .convert('RGB')
            high_1 = high_in[:, :, ::-1].copy()
            high = self.trans(high_1)
            img_name = self.sti_image_filenames[index-len(self.lr_image_filenames)].split('\\')[-1]

            beta = 0.51  #0.1 * random.random() + 0.45
            gamma = 1.78  #0.2 * random.random() + 1.7
            low = beta * torch.pow(high, gamma)
            # low_1 = low_i[:, :, ::-1].copy()
            # low = self.trans(low_1)

        return low, high, img_name

    def __len__(self):
        return self.count

class MemoryFriendlyLoader_test(torch.utils.data.Dataset):
    def __init__(self, img_dir, task):
        self.low_img_dir = img_dir
        self.task = task
        self.train_low_data_names = []
        dir_high = self.low_img_dir + "high"
        dir_low = self.low_img_dir + "low"
        self.hr_image_filenames = [join(dir_high, x) for x in listdir(dir_high)]
        self.lr_image_filenames = [join(dir_low, x) for x in listdir(dir_high)]


        self.count = len(self.lr_image_filenames)

        transform_list = []
        transform_list += [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)
        self.trans = transforms.ToTensor()

    def load_images_transform(self, file):
        im = Image.open(file).convert("HSV")
        img_norm = self.transform(im).numpy()
        img_norm = np.transpose(img_norm, (1, 2, 0))
        # im_hsv = cv2.cvtColor(img_norm, cv2.COLOR_RGB2HSV)
        return img_norm

    def __getitem__(self, index):
        low_i = cv2.imread(self.lr_image_filenames[index])  # .convert('RGB')
        # low_in = cv2.cvtColor(low_i, cv2.COLOR_RGB2HSV)
        low_1 = low_i[:, :, ::-1].copy()
        low = self.trans(low_1)
        high_in = cv2.imread(self.hr_image_filenames[index])  # .convert('RGB')
        # high_in = cv2.cvtColor(high_in, cv2.COLOR_RGB2HSV)
        high_1 = high_in[:, :, ::-1].copy()
        high = self.trans(high_1)

        img_name = self.lr_image_filenames[index].split('\\')[-1]
        return low, high, img_name

    def __len__(self):
        return self.count