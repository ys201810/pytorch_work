# coding=utf-8
import os
import random
import torchtext
from torchtext.vocab import Vectors
from preprocesser import tokenizer_with_preprocessing


def main():
    max_length = 256
    TEXT = torchtext.data.Field(sequential=True,  # データの長さが可変ならTrue
                                tokenize=tokenizer_with_preprocessing,  # 前処理用の関数
                                use_vocab=True,  # ボキャブラリーを作成するか
                                lower=True,  # アルファベットを小文字に変換
                                include_lengths=True,  # 文章の単語数の情報の保持をするか
                                batch_first=True,  # バッチサイズの次元を最初に
                                fix_length=max_length,  # この文字数以上になったらカット、以下なら<pad>を埋める
                                init_token='<cls>',  # 文章の最初に<cls>という文字列をセット
                                eos_token='<eos>')  # 文章の最後に<eos>という文字列をセット
    LABEL = torchtext.data.Field(sequential=False, use_vocab=False)

    data_path = os.path.join('/Users', 'shirai1', 'work', 'pytorch_work', 'pytorch_advanced', '7_nlp_sentiment_transformer', 'data')
    train_val_ds, test_ds = torchtext.data.TabularDataset.splits(
        path=os.path.join(data_path, 'aclImdb'), train='IMDb_train.tsv', test='IMDb_test.tsv', format='tsv',
        fields=[('Text', TEXT), ('Label', LABEL)]
    )

    print('学習・検証用データの長さ:{}'.format(len(train_val_ds)))
    print('学習・検証用データの例:{}'.format(vars(train_val_ds[0])))

    train_ds, val_ds = train_val_ds.split(split_ratio=0.8, random_state=random.seed(1234))
    print('学習用データの長さ:{}'.format(len(train_ds)))
    print('検証用データの長さ:{}'.format(len(val_ds)))
    print('学習用データの例:{}'.format(vars(train_ds[0])))

    english_fasttext_vectors = Vectors(name=os.path.join(data_path, 'wiki-news-300d-1M.vec'))

    print('1単語の次元数:{}'.format(english_fasttext_vectors.dim))
    print('単語数:{}'.format(len(english_fasttext_vectors.itos)))

    TEXT.build_vocab(train_ds, vectors=english_fasttext_vectors, min_freq=10)

    print(TEXT.vocab.vectors.shape)
    print(TEXT.vocab.vectors)
    print(TEXT.vocab.stoi)

    train_dl = torchtext.data.Iterator(train_ds, batch_size=24, train=True)
    val_dl = torchtext.data.Iterator(val_ds, batch_size=24, train=False, sort=False)
    test_dl = torchtext.data.Iterator(test_ds, batch_size=24, sort=False)

    # 動作確認
    batch = next(iter(val_dl))
    print(batch.Text)
    print(batch.Label)


if __name__ == '__main__':
    main()
