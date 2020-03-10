# coding=utf-8
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import models, transforms


class BaseTransform():
    """
    画像をリサイズし、色を標準化する。

    Attributes
    ----------
    resize : int
        リサイズ先の画像の大きさ。
    mean : (R, G, B)
        各色チャネルの平均値。
    std : (R, G, B)
        各色チャネルの標準偏差。
    """
    def __init__(self, resize, mean, std):
        self.base_transform = transforms.Compose([
            transforms.Resize(resize),      # 短い辺の長さがresizeの大きさになる
            transforms.CenterCrop(resize),  # 画像中央をresize × resizeで切り取り
            transforms.ToTensor(),          # Torchテンソルに変換
            transforms.Normalize(mean, std) # 色情報の標準化
        ])

    def __call__(self, img):
        return self.base_transform(img)


class ILSVRCPredictor():
    """
    ILSVRCデータに対するモデルの出力からラベルを求める。

    Attributes
    ----------
    class_index : dictionary
            クラスindexとラベル名を対応させた辞書型変数。
    """

    def __init__(self, class_index):
        self.class_index = class_index

    def predict_max(self, out):
        """
        確率最大のILSVRCのラベル名を取得する。

        Parameters
        ----------
        out : torch.Size([1, 1000])
            Netからの出力。

        Returns
        -------
        predicted_label_name : str
            最も予測確率が高いラベルの名前
        """
        maxid = np.argmax(out.detach().numpy())  # 予測時の最大の値のインデックスを取得
        predicted_label_name = self.class_index[str(maxid)][1]  # インデックスを名称に変換

        return predicted_label_name


def main():
    # VGG16のイメージネットのパラメータロード
    use_pretrained = True
    net = models.vgg16(pretrained=use_pretrained)
    net.eval()  # 推論モード

    image_file = '../../1_image_classification/data/goldenretriever-3724972_640.jpg'
    img = Image.open(image_file)  # PILで画像データを読み込み
    resize = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = BaseTransform(resize, mean, std)
    img_transformed = transform(img)  # PILからtensor型へ

    ILSVRC_class_index = json.load(open('../../1_image_classification/data/imagenet_class_index.json', 'r'))
    predictor = ILSVRCPredictor(ILSVRC_class_index)

    inputs = img_transformed.unsqueeze_(0)  # torch.Size([1, 3, 224, 224])
    out = net(inputs)
    result = predictor.predict_max(out)
    print("入力画像の予測結果：", result)


if __name__ == '__main__':
    main()
