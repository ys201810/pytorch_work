# coding=utf-8
from utils.data_augumentation import Compose, Scale, RandomRotation, RandomMirror, Resize, Normalize_Tensor


class DataTransform:
    def __init__(self, input_size, color_mean, color_std):
        self.data_transform = {
            'train': Compose([
                Scale(scale=[0.5, 1.5]),  # 画像の拡大縮小
                RandomRotation(angle=[-10, 10]),  # 画像の回転
                RandomMirror(),  # 50%の確率で画像の反転
                Resize(input_size),  # リサイズ
                Normalize_Tensor(color_mean, color_std)  # 標準化とTensor型への変換
            ]),
            'val': Compose([
                Resize(input_size),
                Normalize_Tensor(color_mean, color_std)
            ])
        }

    def __call__(self, phase, img, anno_class_img):
        return self.data_transform[phase](img, anno_class_img)

