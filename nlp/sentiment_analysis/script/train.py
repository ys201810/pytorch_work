# coding=utf-8
import torch
from torch import nn, optim
from torch.nn import functional as F
from dataset import get_IMDb_Dataloader_and_text
from models import Embedder, PositionalEncoder, TransformerBlock, TransformerClassification


def highlight(word, attn):
    """Attentionの値が大きいと文字の背景が濃い赤になるhtmlを出力させる関数"""

    html_color = '#%02X%02X%02X' % (
        255, int(255*(1 - attn)), int(255*(1 - attn)))
    return '<span style="background-color: {}"> {}</span>'.format(html_color, word)


def mk_html(index, batch, preds, normlized_weights_1, normlized_weights_2, TEXT):
    """HTMLデータを作成する"""

    # indexの結果を抽出
    sentence = batch.Text[0][index]  # 文章
    label = batch.Label[index]  # ラベル
    pred = preds[index]  # 予測

    # indexのAttentionを抽出と規格化
    attens1 = normlized_weights_1[index, 0, :]  # 0番目の<cls>のAttention
    attens1 /= attens1.max()

    attens2 = normlized_weights_2[index, 0, :]  # 0番目の<cls>のAttention
    attens2 /= attens2.max()

    # ラベルと予測結果を文字に置き換え
    if label == 0:
        label_str = "Negative"
    else:
        label_str = "Positive"

    if pred == 0:
        pred_str = "Negative"
    else:
        pred_str = "Positive"

    # 表示用のHTMLを作成する
    html = '正解ラベル：{}<br>推論ラベル：{}<br><br>'.format(label_str, pred_str)

    # 1段目のAttention
    html += '[TransformerBlockの1段目のAttentionを可視化]<br>'
    for word, attn in zip(sentence, attens1):
        html += highlight(TEXT.vocab.itos[word], attn)
    html += "<br><br>"

    # 2段目のAttention
    html += '[TransformerBlockの2段目のAttentionを可視化]<br>'
    for word, attn in zip(sentence, attens2):
        html += highlight(TEXT.vocab.itos[word], attn)

    html += "<br><br>"

    return html


def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=10):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    torch.backends.cudnn.benchmark = True

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            for batch in dataloaders_dict[phase]:
                inputs = batch.Text[0].to(device)
                labels = batch.Label.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # maskの作成
                    input_pad = 1
                    input_mask = (inputs != input_pad)

                    outputs, _, _ = model(inputs, input_mask)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)  # ラベルの予測

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

            print('Epoch {}/{} | {} | Loss: {} Acc: {}'.format(epoch + 1, num_epochs, phase, epoch_loss, epoch_acc))

    torch.save(model.state_dict(), 'models.pth')
    return model


def main():
    # config setting
    train_dl, val_dl, test_dl, TEXT = get_IMDb_Dataloader_and_text(256, 24)
    model_dim = TEXT.vocab.vectors.shape[1]  # ベクトルの次元数 shape[0]には単語数が入ってる
    max_seq_len = 256
    output_cls_num = 2
    learning_rate = 2e-5
    num_epochs = 10

    dataloaders_dict = {'train': train_dl, 'val': val_dl}

    model = TransformerClassification(text_embedding_vectors=TEXT.vocab.vectors, model_dim=model_dim,
                                      max_seq_len=max_seq_len, output_dim=output_cls_num)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    model.train()

    model.net3_1.apply(weights_init)
    model.net3_2.apply(weights_init)

    print('ネットワークの設定完了')

    # 損失関数の定義
    criterion = nn.CrossEntropyLoss()

    # 最適化関数の定義
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 学習
    trained_model = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)

    # テストデータで検証
    trained_model.eval()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    trained_model.to(device)

    epoch_corrects = 0
    for i, batch in enumerate(test_dl):
        inputs = batch.Text[0].to(device)
        labels = batch.Label.to(device)

        with torch.set_grad_enabled(False):
            input_pad = 1
            input_mask = (inputs != input_pad)

            outputs, normlized_weights_1, normlized_weights_2 = trained_model(inputs, input_mask)
            _, preds = torch.max(outputs, 1)
            index = 3
            html_output = mk_html(index, batch, preds, normlized_weights_1, normlized_weights_2, TEXT)
            with open(str(i) + '.html', 'w') as outf:
                outf.write(html_output)

            epoch_corrects += torch.sum(preds == labels.data)

    epoch_acc = epoch_corrects.double() / len(test_dl.dataset)

    print('テストデータ{}個での正解率:{}'.format(len(test_dl.dataset), epoch_acc))


if __name__ == '__main__':
    main()
