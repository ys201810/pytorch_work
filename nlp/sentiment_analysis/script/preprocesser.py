# coding=utf-8
import re


def preprocessing_text(text):
    text = re.sub('\r', '', text)
    text = re.sub('\n', '', text)  # 改行の削除
    text = re.sub(' ', '', text)   # 半角スペースの削除
    text = re.sub('　', '', text)  # 全角スペースの削除

    text = re.sub(r'[0-9 ０−９]', '0', text)  # 数字を全て0に
    return text


if __name__ == '__main__':
    main()
