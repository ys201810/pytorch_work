# coding=utf-8
import os
import glob
import torch
import torch.utils.data as data
from PIL import Image
from vgg_finetune import ImageTransform
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import tqdm


class HymenopteraDataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)
        img_transformed = self.transform(img, self.phase)

        if self.phase == 'train':
            label = img_path.split('/')[-2]
        else:
            label = img_path.split('/')[-2]

        if label == 'ants':
            label = 0
        elif label == 'bees':
            label = 1

        return img_transformed, label


def make_datapath_list(phase='train'):
    rootpath = os.path.join('../', '..', '1_image_classification', 'data', 'hymenoptera_data')
    target_path = os.path.join(rootpath, phase, '**', '*.jpg')
    image_list = []
    for path in glob.glob(target_path):
        image_list.append(path)

    return image_list


def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
    # epochのループ
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-------------')

        # epochごとの学習と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
            else:
                net.eval()  # モデルを検証モードに

            epoch_loss = 0.0  # epochの損失和
            epoch_corrects = 0  # epochの正解数

            # 未学習時の検証性能を確かめるため、epoch=0の訓練は省略
            if (epoch == 0) and (phase == 'train'):
                continue

            # データローダーからミニバッチを取り出すループ
            for inputs, labels in dataloaders_dict[phase]:

                # optimizerを初期化
                optimizer.zero_grad()

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)  # 損失を計算
                    _, preds = torch.max(outputs, 1)  # ラベルを予測

                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # イタレーション結果の計算
                    # lossの合計を更新
                    epoch_loss += loss.item() * inputs.size(0)
                    # 正解数の合計を更新
                    epoch_corrects += torch.sum(preds == labels.data)

            # epochごとのlossと正解率を表示
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double(
            ) / len(dataloaders_dict[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))


def main():
    train_list = make_datapath_list('train')
    val_list = make_datapath_list('val')
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_dataset = HymenopteraDataset(file_list=train_list, transform=ImageTransform(size, mean, std), phase='train')
    val_dataset = HymenopteraDataset(file_list=val_list, transform=ImageTransform(size, mean, std), phase='val')

    index = 0
    print(train_dataset.__getitem__(index)[0].size())
    print(train_dataset.__getitem__(index)[1])

    # ミニバッチのサイズを指定
    batch_size = 32

    # DataLoaderを作成
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 辞書型変数にまとめる
    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

    # 動作確認
    # batch_iterator = iter(dataloaders_dict["train"])  # イテレータに変換
    # inputs, labels = next(batch_iterator)  # 1番目の要素を取り出す
    # print(inputs.size())
    # print(labels)

    # 学習済みのVGG-16モデルをロード
    # VGG-16モデルのインスタンスを生成
    use_pretrained = True  # 学習済みのパラメータを使用
    net = models.vgg16(pretrained=use_pretrained)

    print(net)
    # VGG16の最後の出力層の出力ユニットをアリとハチの2つに付け替える
    net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

    # 訓練モードに設定
    net.train()
    print('ネットワーク設定完了：学習済みの重みをロードし、訓練モードに設定しました')

    # 損失関数の設定
    criterion = nn.CrossEntropyLoss()

    # 転移学習で学習させるパラメータを、変数params_to_updateに格納する
    params_to_update = []

    # 学習させるパラメータ名
    update_param_names = ["classifier.6.weight", "classifier.6.bias"]

    # 学習させるパラメータ以外は勾配計算をなくし、変化しないように設定
    for name, param in net.named_parameters():
        if name in update_param_names:
            param.requires_grad = True
            params_to_update.append(param)
            print(name)
        else:
            param.requires_grad = False

    # params_to_updateの中身を確認
    print("-----------")
    print(params_to_update)

    # 最適化手法の設定
    optimizer = optim.SGD(params=params_to_update, lr=0.001, momentum=0.9)

    # 学習・検証を実行する
    num_epochs = 2
    train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)

if __name__ == '__main__':
    main()
