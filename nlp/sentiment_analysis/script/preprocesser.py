# coding=utf-8
import re
import string


def preprocessing_text(text):
    text = re.sub('\r', '', text)
    text = re.sub('\n', '', text)  # 改行の削除
    text = re.sub(' ', '', text)   # 半角スペースの削除
    text = re.sub('　', '', text)  # 全角スペースの削除

    text = re.sub(r'[0-9 ０−９]', '0', text)  # 数字を全て0に
    return text


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
    print(tokenizer_with_preprocessing('I like cats.'))


if __name__ == '__main__':
    main()
