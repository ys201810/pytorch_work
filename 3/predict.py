# coding=utf-8
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datasets import make_datapath_list
from preprocess import DataTransform
from network import PSPNet


def main():
    data_root_path = '/Users/shirai1/work/pytorch_work/pytorch_advanced/3_semantic_segmentation/data/VOCdevkit/VOC2012'
    # data_root_path = os.path.join('/home', 'yusuke', 'work', 'data', 'VOCdevkit', 'VOC2012')
    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(data_root_path)

    net = PSPNet(n_classes=21)
    state_dict = torch.load('./weights/pspnet50_30.pth', map_location={'cuda:0': 'cpu'})
    net.load_state_dict(state_dict)

    image_file = './data/cowboy-757575_1280.jpg'
    img = Image.open(image_file)
    img_width, img_height = img.size
    plt.imshow(img)
    plt.show()

    color_mean = (0.485, 0.456, 0.406)
    color_std = (0.229, 0.224, 0.225)
    transform = DataTransform(input_size=475, color_mean=color_mean, color_std=color_std)

    anno_file_path = val_anno_list[0]
    anno_class_img = Image.open(anno_file_path)
    p_palette = anno_class_img.getpalette()
    phase = 'val'
    img, anno_class_img = transform(phase, img, anno_class_img)

    net.eval()
    x = img.unsqueeze(0)
    outputs = net(x)
    y = outputs[0]

    y = y[0].detach().numpy()
    y = np.argmax(y, axis=0)
    anno_class_img = Image.fromarray(np.uint8(y), mode='P')
    anno_class_img = anno_class_img.resize((img_width, img_height), Image.NEAREST)
    anno_class_img.putpalette(p_palette)
    plt.imshow(anno_class_img)
    plt.show()

    trans_img = Image.new('RGBA', anno_class_img.size, (0, 0, 0, 0))
    anno_class_img = anno_class_img.convert('RGBA')

    for x in range(img_width):
        for y in range(img_height):
            pixel = anno_class_img.getpixel((x, y))
            r, g, b, a = pixel

            if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
                continue
            else:
                trans_img.putpixel((x, y), (r, g, b, 150))

    img = Image.open(image_file)
    result = Image.alpha_composite(img.convert('RGBA'), trans_img)
    plt.imshow(result)
    plt.show()

if __name__ == '__main__':
    main()
