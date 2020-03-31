# coding=utf-8
import re
import os
import glob
import MeCab
from gensim.models import word2vec
from torchtext.vocab import Vectors


def wakati_mecab(text, mecab):
    noun_list = []
    wakati_result = mecab.parse(text)
    words = wakati_result.split('\n')
    for ward in words:
        results = ward.split('\t')
        if results[0] == 'EOS' or results[0] == '':
            continue
        if results[3].find('名詞') != -1:
            noun_list.append(results[0])

    return noun_list


def text_preprocess(text):
    delete_charactors = ['\r', '\n', ' ', '　', '"', '#', '$', '%', '&', ',', "'", '\(', '\)', '\*', '\+', ',', '-',
                         '\.', '/', ':', ';', '<', '=', '>', '\?', '@', '\[', '\]', '^', '_', '`', '{', '|', '}', '~', '!',
                         '！', '＃', '＄', '％', '＆', '＼', '’', '（', '）', '＊', '×', '＋', '−', '：', '；', '＜', '＝',
                         '＞', '？', '＠', '「', '」', '＾', '＿', '｀', '『', '』', '【', '】', '｜', '〜', '■️']
    for character in delete_charactors:
        text = re.sub(character, '', text)

    text = re.sub(r'[0-9 ０−９]', '0', text)  # 数字を全て0に
    return text


def make_sentence_list(text_file):
    text_list = []
    mecab = MeCab.Tagger("-Ochasen")
    with open(text_file, 'r') as inf:
        for j, line in enumerate(inf):
            # 最初の2行は記事のURLと作成時刻（？）っぽいので飛ばす
            if j == 0 or j == 1:
                continue
            line = line.rstrip()  # 最後の改行を削除

            replaced_text = text_preprocess(line)
            if not replaced_text:
                continue

            noun_list = wakati_mecab(replaced_text, mecab)
            text_list = text_list + noun_list

    return text_list


def main():
    data_root = os.path.join('/Users', 'shirai1', 'work', 'lstm_work', 'nlp_pytorch', 'classification', 'data', 'text')
    text_files = glob.glob(os.path.join(data_root, 'it-life-hack', 'it-life-hack-*.txt'))

    train_word_list = []

    for i, text_file in enumerate(text_files):
        text_list = make_sentence_list(text_file)
        train_word_list.append(text_list)

    if not os.path.exists('it_life_hach.vec'):
        model = word2vec.Word2Vec(train_word_list, sg=1, size=200, min_count=5, window=5, iter=10)
        model.save("it_life_hach.vec")
    else:
        model = word2vec.Word2Vec.load('it_life_hach.vec')

    print(train_word_list)

    print(model.wv.index2word)

    # w2v = {w: vec for w, vec in zip(model.index2word, model.syn0)}
    print(model.most_similar(positive='PC', topn=10))
    print(model.most_similar(positive='USB', topn=10))
    print(model.most_similar(positive='自作', topn=10))
    print(model.most_similar(positive='動画', topn=10))
    print(model.most_similar(positive='ドワンゴ', topn=10))
    print(model.most_similar(positive='ゲーム', topn=10))
    print(model.most_similar(positive='価格', topn=10))
    print(model.most_similar(negative=['価格'], topn=10))
    print(model.most_similar(positive=['YouTube', 'niconico'], negative=['再生'], topn=10))


if __name__ == '__main__':
    main()
