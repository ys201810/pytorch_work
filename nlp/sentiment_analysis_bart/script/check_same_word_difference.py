# coding=utf-8
import os
import torch
import collections
import numpy as np
from attrdict import AttrDict
from nlp.sentiment_analysis_bart.script.models import BertModel
from nlp.sentiment_analysis_bart.script.tokenizer import BasicTokenizer, WordpieceTokenizer
import torch.nn.functional as F

class BertTokenizer(object):
    """ BERT用のTokenizer """
    def __init__(self, vocab_file, do_lower_case):
        self.vocab, self.ids_to_tokens = load_vocab(vocab_file)

        never_split = ('[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]')  # これらの単語はTokenizeしない

        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case, never_split=never_split)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        """ textをBERT用の単語に分割 """
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """ 分割した単語列をID列に変換 """
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])

        return ids

    def convert_ids_to_tokens(self, ids):
        """ ID列を単語列に変換 """
        tokens = []
        for id in ids:
            tokens.append(self.ids_to_tokens[id])
        return tokens


def load_vocab(vocab_file):
    vocab = collections.OrderedDict()
    ids_to_tokens = collections.OrderedDict()
    index = 0
    with open(vocab_file, 'r', encoding='utf-8') as inf:
        while True:
            token = inf.readline()
            if not token:
                break
            token = token.strip()

            vocab[token] = index
            ids_to_tokens[index] = token
            index += 1
    return vocab, ids_to_tokens


def main():
    config = {
        'attention_probs_dropout_prob': 0.1,
        'hidden_act': 'gelu',
        'hidden_dropout_prob': 0.1,
        'hidden_size': 768,
        'initializer_range': 0.02,
        'intermediate_size': 3072,
        'max_position_embeddings': 512,
        'num_attention_heads': 12,
        'num_hidden_layers': 12,
        'type_vocab_size': 2,
        'vocab_size': 30522
    }
    config = AttrDict(config)

    net = BertModel(config)
    net.eval()
    param_names = []

    # 作成したネットワークの名前の表示とlistへのappend
    for name, param in net.named_parameters():
        print(name)
        param_names.append(name)

    # ネットワークの各層の名前を保持
    new_state_dict = net.state_dict().copy()

    # pre-trainモデルをloadして各層の名前を自前ネットワークの名前に変えてload
    # https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz
    weight_path = os.path.join('../', '..', '..', '..', '8_nlp_sentiment_bert', 'weights', 'pytorch_model.bin')
    load_state_dict = torch.load(weight_path)
    for index, (key_name, value) in enumerate(load_state_dict.items()):
        name = param_names[index]
        new_state_dict[name] = value
        print('{} -> {}'.format(key_name, name))

        if index + 1 >= len(param_names):
            break

    net.load_state_dict(new_state_dict)

    # Bert用のTokenizerの用意
    vocab_file = os.path.join('../', '..', '..', '..', '8_nlp_sentiment_bert', 'vocab', 'bert-base-uncased-vocab.txt')
    tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=True)

    text1 = '[CLS] I accessed the bank account. [SEP]'
    text2 = '[CLS] He transferred the deposit money into the bank account. [SEP]'
    text3 = '[CLS] We play soccer at the bank of the river. [SEP]'

    # 文章を単語に分割
    tokenized_text1 = tokenizer.tokenize(text1)
    tokenized_text2 = tokenizer.tokenize(text2)
    tokenized_text3 = tokenizer.tokenize(text3)

    # 単語をIDに変換
    indexed_tokens1 = tokenizer.convert_tokens_to_ids(tokenized_text1)
    indexed_tokens2 = tokenizer.convert_tokens_to_ids(tokenized_text2)
    indexed_tokens3 = tokenizer.convert_tokens_to_ids(tokenized_text3)

    # bankの単語の出現位置
    bank_position1 = np.where(np.array(tokenized_text1) == 'bank')[0][0]
    bank_position2 = np.where(np.array(tokenized_text2) == 'bank')[0][0]
    bank_position3 = np.where(np.array(tokenized_text3) == 'bank')[0][0]

    # 単語のID列をテンソルに
    tokens_tensor1 = torch.tensor([indexed_tokens1])
    tokens_tensor2 = torch.tensor([indexed_tokens2])
    tokens_tensor3 = torch.tensor([indexed_tokens3])

    bank_word_id = tokenizer.convert_tokens_to_ids(['bank'])[0]
    print(tokens_tensor1)  # text1の単語のIDを出力

    with torch.no_grad():
        encoded_layers1, _ = net(tokens_tensor1, output_all_encoded_layers=True)
        encoded_layers2, _ = net(tokens_tensor2, output_all_encoded_layers=True)
        encoded_layers3, _ = net(tokens_tensor3, output_all_encoded_layers=True)

    bank_vector0 = net.embeddings.word_embeddings.weight[bank_word_id]
    bank_vector1_1 = encoded_layers1[0][0, bank_position1]
    bank_vector1_12 = encoded_layers1[11][0, bank_position1]
    bank_vector2_1 = encoded_layers2[0][0, bank_position2]
    bank_vector2_12 = encoded_layers2[11][0, bank_position2]
    bank_vector3_1 = encoded_layers3[0][0, bank_position3]
    bank_vector3_12 = encoded_layers3[11][0, bank_position3]

    print('bankの初期ベクトルと文章1の1段目のbankの類似度:{}'.format(F.cosine_similarity(bank_vector0, bank_vector1_1, dim=0)))
    print('bankの初期ベクトルと文章1の12段目のbankの類似度:{}'.format(F.cosine_similarity(bank_vector0, bank_vector1_12, dim=0)))
    print('bankの初期ベクトルと文章2の1段目のbankの類似度:{}'.format(F.cosine_similarity(bank_vector1_1, bank_vector2_1, dim=0)))
    print('bankの初期ベクトルと文章3の1段目のbankの類似度:{}'.format(F.cosine_similarity(bank_vector1_1, bank_vector3_1, dim=0)))
    print('bankの初期ベクトルと文章2の12段目のbankの類似度:{}'.format(F.cosine_similarity(bank_vector1_12, bank_vector2_12, dim=0)))
    print('bankの初期ベクトルと文章3の12段目のbankの類似度:{}'.format(F.cosine_similarity(bank_vector1_12, bank_vector3_12, dim=0)))


if __name__ == '__main__':
    main()
