# coding=utf-8
import os
import math
import torch
from torch import nn
from torch.utils import data
from datasets import VOCDataset, make_datapath_list
from preprocess import DataTransform
from network import PSPNet
from loss import PSPLoss
from torch import optim


def main():
    batch_size = 8
    color_mean = (0.485, 0.456, 0.406)
    color_std = (0.229, 0.224, 0.225)

    # data_root_path = os.path.join('/Users', 'shirai1', 'work', 'pytorch_work', 'pytorch_advanced',
    #                               '3_semantic_segmentation', 'data', 'VOCdevkit', 'VOC2012')
    data_root_path = os.path.join('/home', 'yusuke', 'work', 'data', 'VOCdevkit', 'VOC2012')
    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(data_root_path)
    train_dataset = VOCDataset(train_img_list, train_anno_list, phase='train',
                               transform=DataTransform(input_size=475, color_mean=color_mean, color_std=color_std))
    val_dataset = VOCDataset(val_img_list, val_anno_list, phase='val',
                               transform=DataTransform(input_size=475, color_mean=color_mean, color_std=color_std))

    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader}

    net = PSPNet(n_classes=150)
    state_dict = net.load_state_dict(torch.load('./weights/pspnet50_ADE20K.pth'))

    n_classes = 21
    net.decode_feature.classification = nn.Conv2d(in_channels=512, out_channels=n_classes, kernel_size=1, stride=1,
                                                  padding=0)
    net.aux.classification = nn.Conv2d(in_channels=256, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    net.decode_feature.classification.apply(weights_init)
    net.aux.classification.apply(weights_init)

    print('ネットワークの設定完了。学習済みの重みのロード終了')

    # optimizerをネットワークの層の名前ごとにlrを変えて定義
    optimizer = optim.SGD([
        {'params': net.feature_conv.parameters(), 'lr': 1e-3},
        {'params': net.feature_res_1.parameters(), 'lr': 1e-3},
        {'params': net.feature_res_2.parameters(), 'lr': 1e-3},
        {'params': net.feature_dilated_res_1.parameters(), 'lr': 1e-3},
        {'params': net.feature_dilated_res_2.parameters(), 'lr': 1e-3},
        {'params': net.pyramid_pooling.parameters(), 'lr': 1e-3},
        {'params': net.decode_feature.parameters(), 'lr': 1e-2},
        {'params': net.aux.parameters(), 'lr': 1e-2},
    ], momentum=0.9, weight_decay=0.0001)

    # スケジューラーの設定
    def lambda_epoch(epoch):
        max_epoch = 30
        return math.pow((1 - epoch / max_epoch), 0.9)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch)

    


if __name__ == '__main__':
    main()
