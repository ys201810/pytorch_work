# coding=utf-8
import os
import re
import json
import torch
import string
import torchtext
from nlp.sentiment_analysis_bart.script.models import BertModel, set_learned_params, BertForIMBd
from nlp.sentiment_analysis_bart.script.tokenizer import BertTokenizer, load_vocab
from torch import nn, optim
import random
from attrdict import AttrDict


"""
def highlight(word, attn):
    # Attentionの値が大きいと文字の背景が濃い赤になるhtmlを出力させる関数

    html_color = '#%02X%02X%02X' % (
        255, int(255*(1 - attn)), int(255*(1 - attn)))
    return '<span style="background-color: {}"> {}</span>'.format(html_color, word)


def mk_html(index, batch, preds, normlized_weights_1, normlized_weights_2, TEXT):
    # HTMLデータを作成する

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
"""


def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=10):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    torch.backends.cudnn.benchmark = True
    batch_size = dataloaders_dict['train'].batch_size

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_corrects = 0
            iteration = 1

            for batch in dataloaders_dict[phase]:
                inputs = batch.Text[0].to(device)
                labels = batch.Label.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs, token_type_ids=None, attention_mask=None, output_all_encoded_layers=False, attention_show_flg=False)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)  # ラベルの予測

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        if iteration % 10 == 0:
                            acc = torch.sum(preds == labels.data).double() / batch_size
                            print('イテレーション:{} || Loss: {:.4f} || 正解数:{}'.format(iteration, loss.item(), acc))

                    iteration += 1

                    epoch_loss += loss.item() * batch_size
                    epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

            print('Epoch {}/{} | {} | Loss: {} Acc: {}'.format(epoch + 1, num_epochs, phase, epoch_loss, epoch_acc))

    return model


def preprocessing_text(text):
    """ 改行コードの削除、カンマ・ピリオド以外の記号のスペース置換、カンマ・ピリオドの前後にスペースを入れる """
    text = re.sub('<br />', '', text)

    # 記号のスペース置換
    for p in string.punctuation:
        if p == '.' or p == ',':
            continue
        else:
            text = text.replace(p, ' ')

    text = text.replace('.', ' . ')
    text = text.replace(',', ' , ')
    return text


def tokenizer_with_preprocessing(text):
    # 単語分割用のvertのtokenizerの用意
    # data_path = os.path.join('/Users', 'shirai1', 'work', 'pytorch_work', 'pytorch_advanced', '8_nlp_sentiment_bert')
    data_path = os.path.join('/home', 'yusuke', 'work', 'pytorch_work', 'nlp', 'sentiment_analysis_bart', 'data', '8_nlp_sentiment_bert')

    bert_vocab_file = os.path.join(data_path, 'vocab', 'bert-base-uncased-vocab.txt')
    bert_tokenizer = BertTokenizer(vocab_file=bert_vocab_file, do_lower_case=True)

    text = preprocessing_text(text)
    result = bert_tokenizer.tokenize(text)
    return result


def main():
    # config setting
    max_length = 256
    batch_size = 32
    num_epochs = 3
    # data_path = os.path.join('/Users', 'shirai1', 'work', 'pytorch_work', 'pytorch_advanced', '8_nlp_sentiment_bert')
    data_path = os.path.join('/home', 'yusuke', 'work', 'pytorch_work', 'nlp', 'sentiment_analysis_bart', 'data', '8_nlp_sentiment_bert')

    bert_vocab_file = os.path.join(data_path, 'vocab', 'bert-base-uncased-vocab.txt')
    bert_tokenizer = BertTokenizer(vocab_file=bert_vocab_file, do_lower_case=True)

    # Pytorchでテキストを扱うためのデータセットの設定
    TEXT = torchtext.data.Field(sequential=True,                        # データの長さが可変である時True
                                tokenize=tokenizer_with_preprocessing,  # 文章を読み込んだ時に処理する前処理関数の指定
                                use_vocab=True,                         # 単語をボキャブラリーに追加するか
                                lower=True,                             # アルファベットを小文字に変換するか
                                include_lengths=True,                   # 文章の単語数のデータを保持するか
                                batch_first=True,                       # バッチサイズをテンソルの最初の次元で扱う
                                fix_length=max_length,                  # 指定した数になるようにPADDINGする
                                init_token='[CLS]',                     # 文頭
                                eos_token='[SEP]',                      # 文末
                                pad_token='[PAD]',                      # padding
                                unk_token='[UNK]')                      # 未知語
    LABEL = torchtext.data.Field(sequential=False, use_vocab=False)

    train_val_ds, test_ds = torchtext.data.TabularDataset.splits(
        path=os.path.join(data_path, 'data'), train='IMDb_train.tsv', test='IMDb_test.tsv', format='tsv',
        fields=[('Text', TEXT), ('Label', LABEL)])

    # train_val_ds, test_ds = torchtext.data.TabularDataset.splits(
    #     path=os.path.join(data_path, 'data'),
    #     train='IMDb_train.tsv', test='IMDb_test.tsv', format='tsv', fields=[('Text', TEXT), ('Label', LABEL)])
    train_ds, val_ds = train_val_ds.split(split_ratio=0.8, random_state=random.seed(1234))

    # TEXTでvocab関数が利用できるようにbuildをするが、今回利用したいのはプレトレインのvocabなのでbuildするが、プレトレインで再度上書きする。
    vocab_vert, ids_to_tokens_bert = load_vocab(vocab_file=os.path.join(data_path, 'vocab', 'bert-base-uncased-vocab.txt'))

    TEXT.build_vocab(train_ds, min_freq=1)
    TEXT.vocab.stoi = vocab_vert

    # データローダーの用意
    train_dl = torchtext.data.Iterator(
        train_ds, batch_size=batch_size, train=True)

    val_dl = torchtext.data.Iterator(
        val_ds, batch_size=batch_size, train=False, sort=False)

    test_dl = torchtext.data.Iterator(
        test_ds, batch_size=batch_size, train=False, sort=False)

    dataloaders_dict = {'train': train_dl, 'val': val_dl}

    # 動作確認
    batch = next(iter(val_dl))
    print(batch.Text)
    print(batch.Label)

    text_minibatch_1 = (batch.Text[0][1]).numpy()
    text = bert_tokenizer.convert_ids_to_tokens(text_minibatch_1)
    print(text)

    # BERTのコンフィグファイルからモデルを作成
    conf_file = os.path.join(data_path, 'weights', 'bert_config.json')
    json_file = open(conf_file, 'r')
    json_object = json.load(json_file)

    config = AttrDict(json_object)
    bert_net = BertModel(config)

    # プレトレインのロード
    bert_net = set_learned_params(bert_net, weights_path=os.path.join(data_path, 'weights', 'pytorch_model.bin'))

    # IMDb向けに最後に全結合層を追加したネットワークを作成
    net = BertForIMBd(bert_net)
    net.train()
    print('ネットワーク設定完了')

    # IMBd向けのファインチューニングで最終層以外を学習しないようにする
    for name, param in net.named_parameters():
        param.requires_grad = False
    for name, param in net.bert.encoder.layer[-1].named_parameters():
        param.requires_grad = True
    for name, param in net.cls.named_parameters():
        param.requires_grad = True

    # 最適化関数の定義
    optimizer = optim.Adam([
        {'params': net.bert.encoder.layer[-1].parameters(), 'lr': 5e-5},
        {'params': net.cls.parameters(), 'lr': 5e-5}
    ], betas=(0.9, 0.999))

    # 損失関数の定義
    criterion = nn.CrossEntropyLoss()

    # 学習
    trained_model = train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)

    # 学習済みモデルの保存
    torch.save(trained_model.state_dict(), 'models.pth')

    # テストデータでの予測
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    trained_model.eval()
    trained_model.to(device)

    epoch_corrects = 0

    for batch in test_dl:
        inputs = batch.Text[0].to(device)
        labels = batch.Label.to(device)

        with torch.set_grad_enabled(False):
            outputs = trained_model(inputs, token_type_ids=None, attention_mask=None, output_all_encoded_layers=False, attention_show_flg=False)

            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            epoch_corrects += torch.sum(preds == labels.data)

    epoch_acc = epoch_corrects.double() / len(test_dl.dataset)

    print('テストデータ:{}個での正解率:{:.4}'.format(len(test_dl.dataset), epoch_acc))


if __name__ == '__main__':
    main()
