# coding=utf-8
import re
import os
import glob
import MeCab


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

            print('整形前:{}'.format(line))
            print('整形後:{}'.format(replaced_text))

            noun_list = wakati_mecab(replaced_text, mecab)
            text_list = text_list + noun_list

    return text_list


def main():
    data_root = os.path.join('/Users', 'shirai1', 'work', 'lstm_work', 'nlp_pytorch', 'classification', 'data', 'text')
    text_files = glob.glob(os.path.join(data_root, 'it-life-hack', 'it-life-hack-*.txt'))

    train_word_list = []

    for i, text_file in enumerate(text_files):
        print(text_file)
        text_list = make_sentence_list(text_file)
        print(text_list)
        train_word_list.append(text_list)
        print(train_word_list)
        if i == 1:
            exit(1)

if __name__ == '__main__':
    main()
