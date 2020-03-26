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
                    if text_file.find('pos') > 0:
                        text = '\t'.join([text, '1', '\n'])
                    else:
                        text = '\t'.join([text, '0', '\n'])
                    outf.write(text)


def main():
    data_path = os.path.join('/Users', 'shirai1', 'work', 'pytorch_work', 'pytorch_advanced',
                             '7_nlp_sentiment_transformer', 'data', 'aclImdb')
    if os.path.exists(os.path.join(data_path, 'IMDb_train.tsv')):
        make_tsv_file(data_path, 'train')
    if os.path.exists(os.path.join(data_path, 'IMDb_test.tsv')):
        make_tsv_file(data_path, 'test')


if __name__ == '__main__':
    main()
