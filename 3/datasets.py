# coding=utf-8
import os
from torch.utils import data
from PIL import Image


class VOCDataset(data.Dataset):
    def __init__(self, img_list, anno_list, phase, transform):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img, anno_class_img = self.pull_item(index)
        return img, anno_class_img

    def pull_item(self, index):
        image_file_path = self.img_list[index]
        img = Image.open(image_file_path)

        anno_file_path = self.anno_list[index]
        anno_class_img = Image.open(anno_file_path)

        img, anno_class_img = self.transform(self.phase, img, anno_class_img)

        return img, anno_class_img


def make_datapath_list(root_path):
    img_path_template = os.path.join(root_path, 'JPEGImages', '{}.jpg')
    anno_path_template = os.path.join(root_path, 'SegmentationClass', '{}.png')

    train_id_names = os.path.join(root_path, 'ImageSets', 'Segmentation', 'train.txt')
    val_id_names = os.path.join(root_path, 'ImageSets', 'Segmentation', 'val.txt')

    train_img_list, train_anno_list = [], []

    with open(train_id_names, 'r') as inf:
        for line in inf:
            file_id = line.strip()
            img_path = img_path_template.format(file_id)
            anno_path = anno_path_template.format(file_id)
            train_img_list.append(img_path)
            train_anno_list.append(anno_path)

    val_img_list, val_anno_list = [], []

    with open(val_id_names, 'r') as inf:
        for line in inf:
            file_id = line.strip()
            img_path = img_path_template.format(file_id)
            anno_path = anno_path_template.format(file_id)
            val_img_list.append(img_path)
            val_anno_list.append(anno_path)

    return train_img_list, train_anno_list, val_img_list, val_anno_list


def main():
    pass


if __name__ == '__main__':
    main()
