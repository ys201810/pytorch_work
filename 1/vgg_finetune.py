# coding=utf-8
import glob
import os
import random
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms


class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            "train": transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
                ]),
            "val": transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])}

    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)


def main():
    # 乱数のシードの設定
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)




if __name__ == '__main__':
    main()
