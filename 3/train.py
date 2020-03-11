# coding=utf-8
import os
from torch.utils import data
from datasets import VOCDataset, make_datapath_list
from preprocess import DataTransform


def main():
    batch_size = 8
    color_mean = (0.485, 0.456, 0.406)
    color_std = (0.229, 0.224, 0.225)

    data_root_path = os.path.join('')
    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(data_root_path)
    train_dataset = VOCDataset(train_img_list, train_anno_list, phase='train',
                               transform=DataTransform(input_size=475, color_mean=color_mean, color_std=color_std))
    val_dataset = VOCDataset(val_img_list, val_anno_list, phase='val',
                               transform=DataTransform(input_size=475, color_mean=color_mean, color_std=color_std))

    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader}

    # 動作確認
    batch_iterator = iter(dataloaders_dict['val'])
    images, anno_clss_images = next(batch_iterator)
    print(images.size, anno_clss_images.size)

    

if __name__ == '__main__':
    main()
