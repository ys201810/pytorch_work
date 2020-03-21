# coding=utf-8
from tokenizer import janome_tokenize, mecab_tokenize
from preprocesser import preprocessing_text
import torchtext
from torchtext.vocab import Vectors


def tokenizer_with_preprocessing(text):
    text = preprocessing_text(text)
    result = mecab_tokenize(text)

    return result


def save_word2vec_model():
    # http://www.cl.ecei.tohoku.ac.jp/~m-suzuki/jawiki_vector/data/20170201.tar.bz2を解凍して得たbinファイル。
    from gensim.models import KeyedVectors
    model = KeyedVectors.load_word2vec_format('entity_vector.model.bin', binary=True)
    model.wv.save_word2vec_format('../data/japanese_word2vec_vectors.vec')


def main():
    text = '昨日は とても暑く、気温が37度もあった'
    print(tokenizer_with_preprocessing(text))

    max_length = 25
    TEXT = torchtext.data.Field(sequential=True,  # データの長さが可変かどうか。
                                tokenize=tokenizer_with_preprocessing,  # 前処理として適応する関数
                                use_vocab=True,  # 単語をボキャブラリーに追加するかどうか
                                lower=True,  # アルファベットを小文字にするか
                                include_lengths=True,  # 文章の単語数のデータを保持するか
                                batch_first=True,  # バッチサイズをTensorの次元の先頭にするか
                                fix_length=max_length)  # 全部の文章が同じ長さになる様にpaddingするサイズ
    LABEL = torchtext.data.Field(sequential=False,  # データの長さは固定
                                 use_vocab=False)  # ボキャブラリーに追加しない

    # データのロード(torch.data.datasetの様なDataLoaderに食わすデータセットの作成)
    train_ds, val_ds, test_ds = torchtext.data.TabularDataset.splits(
        path='../data/', train='text_train.tsv', validation='text_val.tsv', test='text_test.tsv',
        fields=[('Text', TEXT), ('Label', LABEL)], format='tsv')
    print('tain_num:{}'.format(len(train_ds)))
    print('train_example1:{}'.format(vars(train_ds[0])))
    print('train_example2:{}'.format(vars(train_ds[1])))

    # 単語を数値化するためのボキャブラリーの作成
    TEXT.build_vocab(train_ds, min_freq=1)  # min_freqは最小出現回数
    print(TEXT.vocab.freqs)  # 単語と出現回数のdictの様なもの
    print(TEXT.vocab.stoi)  # string to ID 単語とIDのdictの様なもの

    # データローダーの作成
    train_dl = torchtext.data.Iterator(train_ds, batch_size=2, train=True)
    val_dl = torchtext.data.Iterator(val_ds, batch_size=2, train=False, sort=False)
    test_dl = torchtext.data.Iterator(test_ds, batch_size=2, train=False, sort=False)

    # 動作確認
    batch = next(iter(val_dl))
    print(batch.Text, batch.Label)  # batch.Textはtuple。 1つ目にTensorが2つ目が出現単語数。 batch.LabelはTensor。
    print(batch.Text[0].shape, batch.Label.shape)  # torch.Size([2, 25]) -> batch_size分25の次元。 torch.Size([])

    # word2vecでの単語のベクトル化
    japanese_word2vec_vectors = Vectors(name='../../../../7_nlp_sentiment_transformer/data/entity_vector/'
                                             'japanese_word2vec_vectors.vec')

    print('1単語を表現する次元数:{}'.format(japanese_word2vec_vectors.dim))  # 200次元
    print('単語数:{}'.format(len(japanese_word2vec_vectors.itos)))        # 1015474個

    TEXT.build_vocab(train_ds, vectors=japanese_word2vec_vectors, min_freq=1)
    print(TEXT.vocab.vectors.shape)  # torch.Size([51, 200])
    print(TEXT.vocab.stoi)

    import torch.nn.functional as F
    tensor_calc = TEXT.vocab.vectors[43] - TEXT.vocab.vectors[40] + TEXT.vocab.vectors[48]  # 姫 - 女性 + 男性
    print('女王と計算したベクトルとのコサイン類似度:{}'.format(F.cosine_similarity(tensor_calc, TEXT.vocab.vectors[41], dim=0)))
    print('王と計算したベクトルとのコサイン類似度:{}'.format(F.cosine_similarity(tensor_calc, TEXT.vocab.vectors[46], dim=0)))
    print('王子と計算したベクトルとのコサイン類似度:{}'.format(F.cosine_similarity(tensor_calc, TEXT.vocab.vectors[47], dim=0)))
    print('機械学習と計算したベクトルとのコサイン類似度:{}'.format(F.cosine_similarity(tensor_calc, TEXT.vocab.vectors[45], dim=0)))

    # fasttextでの単語のベクトル化
    japanese_fasttext_vectors = Vectors(name='../../../../7_nlp_sentiment_transformer/data/fasttext_model.vec')

    print('1単語を表現する次元数:{}'.format(japanese_fasttext_vectors.dim))
    print('単語数:{}'.format(len(japanese_fasttext_vectors.itos)))

    TEXT.build_vocab(train_ds, vectors=japanese_fasttext_vectors, min_freq=1)
    print(TEXT.vocab.vectors.shape)  # torch.Size([51, 300])
    print(TEXT.vocab.stoi)

    import torch.nn.functional as F
    tensor_calc_ft = TEXT.vocab.vectors[43] - TEXT.vocab.vectors[40] + TEXT.vocab.vectors[48]  # 姫 - 女性 + 男性
    print('女王と計算したベクトルとのコサイン類似度:{}'.format(F.cosine_similarity(tensor_calc_ft, TEXT.vocab.vectors[41], dim=0)))
    print('王と計算したベクトルとのコサイン類似度:{}'.format(F.cosine_similarity(tensor_calc_ft, TEXT.vocab.vectors[46], dim=0)))
    print('王子と計算したベクトルとのコサイン類似度:{}'.format(F.cosine_similarity(tensor_calc_ft, TEXT.vocab.vectors[47], dim=0)))
    print('機械学習と計算したベクトルとのコサイン類似度:{}'.format(F.cosine_similarity(tensor_calc_ft, TEXT.vocab.vectors[45], dim=0)))

if __name__ == '__main__':
    main()
