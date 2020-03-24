# coding=utf-8
"""
IMDbデータセット(http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)を使って、
positiveな評価のレビューとnegativeな評価のレビューの学習用のデータを作成する。
"""
import glob
import os
import re
import string


def make_tsv_file(data_path, kind):
    if os.path.exists(os.path.join(data_path, 'IMDb_' + kind + '.tsv')):
        os.remove(os.path.join(data_path, 'IMDb_' + kind + '.tsv'))

    pos_data_path = os.path.join(data_path, kind, 'pos')
    pos_files = glob.glob(os.path.join(pos_data_path, '*.txt'))
    neg_data_path = os.path.join(data_path, kind, 'neg')
    neg_files = glob.glob(os.path.join(neg_data_path, '*.txt'))

    with open(os.path.join(data_path, 'IMDb_' + kind + '.tsv'), 'a') as outf:
        for target_files in (pos_files, neg_files):
            for text_file in target_files:
                with open(text_file, 'r', encoding='utf-8') as inf:
                    text = inf.readline()
                    text = text.replace('\t', ' ')  # tsvにしたいので先にタブを半角スペースに変換。
                    text = '\t'.join([text, '1', '\n'])
                    outf.write(text)


def preprocessing_text(text):
    text = re.sub('<br />', '', text)

    for symbol in string.punctuation:
        if symbol != '.' and symbol != ',':  # .と,以外の記号を削除する。
            text = text.replace(symbol, '')

    text = text.replace('.', ' . ')  # .は前後に半角スペースを入れることで、一つの単語的に扱う。
    text = text.replace(',', ' , ')  # ,も同上  (これをしないと、.が付いた単語が別の単語と扱われてしまうから)
    return text


def tokunizer_punctuation(text):
    return text.strip().split()  # 記号を半角スペースに置き換えているので、前後の半角スペースをstripで削除して半角スペースで単語リスト作成


def tokenizer_with_preprocessing(text):
    text = preprocessing_text(text)
    results = tokunizer_punctuation(text)
    return results


def main():
    data_path = os.path.join('/Users', 'shirai1', 'work', 'pytorch_work', 'pytorch_advanced',
                             '7_nlp_sentiment_transformer', 'data', 'aclImdb')
    if not os.path.exists(os.path.join(data_path, 'IMDb_train.tsv')):
        make_tsv_file(data_path, 'train')
    if not os.path.exists(os.path.join(data_path, 'IMDb_test.tsv')):
        make_tsv_file(data_path, 'test')

    print(tokenizer_with_preprocessing('I like cats.'))



if __name__ == '__main__':
    main()
