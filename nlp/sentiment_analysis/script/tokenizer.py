# coding=utf-8
from janome.tokenizer import Tokenizer
import MeCab


def janome_tokenize(text, wakati=True):
    j_t = Tokenizer()

    return [tok for tok in j_t.tokenize(text, wakati=wakati)]


def mecab_tokenize(text):
    m_t = MeCab.Tagger('-Ochasen -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
    return [word.split('\t')[0] for word in m_t.parse(text).split('\n')]


if __name__ == '__main__':
    print(janome_tokenize('私は機械学習が好き。'))
    print(mecab_tokenize('私は機械学習が好き。')[0:-2])
