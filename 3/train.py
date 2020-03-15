# coding=utf-8
import os
import time
import math
import torch
import pandas as pd
from torch import nn
from torch.utils import data
from datasets import VOCDataset, make_datapath_list
from preprocess import DataTransform
from network import PSPNet
from loss import PSPLoss
from torch import optim


def train_model(net, dataloaders_dict, criterion, scheduler, optimizer, num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('使用デバイス:{}'.format(device))

    net.to(device)
    torch.backends.cudnn.benchmark = True  # ネットワークの学習がある程度固定なら、これをTrueに

    num_train_imgs = len(dataloaders_dict['train'].dataset)
    num_val_imgs = len(dataloaders_dict['val'].dataset)
    batch_size = dataloaders_dict['train'].batch_size

    iteration = 1
    logs = []

    batch_multiplier = 3

    for epoch in range(num_epochs):
        t_epoch_start = time.time()
        t_iter_start = time.time()
        epoch_train_loss, epoch_val_loss = 0.0, 0.0

        print('Epoch {}/{}'.format((epoch + 1), num_epochs))

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを学習モード[パラメータの更新有り]にセット
                scheduler.step()  # schedulerの更新
                optimizer.zero_grad()  # 1回前のiterationで学習した勾配のリセット
                print('[train]')
            else:
                if (epoch + 1) / 5 == 0:  # evalは5epochに1回
                    net.eval()  # モデルを検証モード[パラメータの更新無し]にセット
                    print('[val]')
                else:
                    continue

            count = 0
            for images, anno_class_images in dataloaders_dict[phase]:
                if images.size()[0] == 1:  # バッチサイズが1だとBatchNormでエラーになるのでcontinueする
                    continue

                images, anno_class_images = images.to(device), anno_class_images.to(device)

                if phase == 'train' and count == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    count = batch_multiplier

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(images)
                    loss = criterion(outputs, anno_class_images.long()) / batch_multiplier

                    if phase == 'train':
                        loss.backward()
                        count -= 1

                        if iteration % 10 == 0:
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            print('イテレーション {} || Loss:{} || 10iter:{} sec.'
                                  .format(iteration, loss.item() / batch_size * batch_multiplier, duration))
                            t_iter_start = time.time()
                        epoch_train_loss += loss.item() * batch_multiplier
                        iteration += 1
                    else:
                        epoch_val_loss += loss.item() * batch_multiplier

        t_epoch_finish = time.time()
        print('epoch {} || Epoch_TRAIN_Loss:{} || Epoch_VAL_Loss:{}'.format(
            epoch + 1, epoch_train_loss / num_train_imgs, epoch_val_loss / num_val_imgs
        ))
        print('timer: {} sec'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

        log_epoch = {'epoch': epoch+1, 'train_loss': epoch_train_loss / num_train_imgs,
                     'val_loss': epoch_val_loss / num_val_imgs}
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv("log_output.csv")

        torch.save(net.state_dict(), 'weghts/pspnet50_' + str(epoch + 1) + '.pth')

def main():
    batch_size = 4
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

    criterion = PSPLoss(aux_weight=0.4)

    epochs = 30

    train_model(net, dataloaders_dict, criterion, scheduler, optimizer, num_epochs=epochs)

if __name__ == '__main__':
    main()
