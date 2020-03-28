# coding=utf-8
import math
import torch
from torch import nn
from torch.nn import functional as F


class Embedder(nn.Module):
    """ IDで示される単語をベクトルに変換する"""
    def __init__(self, text_embedding_vectors):
        super(Embedder, self).__init__()

        self.embeddings = nn.Embedding.from_pretrained(embeddings=text_embedding_vectors, freeze=True)  # freeze: バックプロップでの更新なし

    def forward(self, x):
        x_vec = self.embeddings(x)
        return x_vec


class PositionalEncoder(nn.Module):
    """ 入力された単語の位置を示すベクトル情報を不可する """
    def __init__(self, model_dim=300, max_seq_len=256):
        super().__init__()
        self.model_dim = model_dim
        pe = torch.zeros(max_seq_len, model_dim)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        pe = pe.to(device)

        for pos in range(max_seq_len):
            for i in range(0, model_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / model_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / model_dim)))

        self.pe = pe.unsqueeze(0)  # バッチサイズの次元を追加
        self.pe.requires_grad = False  # 勾配の計算なし

    def forward(self, x):
        result = math.sqrt(self.model_dim) * x + self.pe
        return result


class Attention(nn.Module):
    """ 各単語ごとの関係性を踏まえた特徴量抽出 """
    def __init__(self, model_dim=300):
        super().__init__()

        self.q_linear = nn.Linear(model_dim, model_dim)
        self.v_linear = nn.Linear(model_dim, model_dim)
        self.k_linear = nn.Linear(model_dim, model_dim)

        self.out = nn.Linear(model_dim, model_dim)
        self.d_k = model_dim

    def forward(self, q, k, v, mask):
        k = self.k_linear(k)
        q = self.q_linear(q)
        v = self.v_linear(v)

        # Attentionの計算。 各値をそのまま足すと大きくなりすぎるので、root(model_dim)で割って調整
        weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_k)

        mask = mask.unsqueeze(1)
        weights = weights.masked_fill(mask == 0, -1e9)  # maskが0(<pad>)の箇所にマイナス無限大をセット

        normalized_weights = F.softmax(weights, dim=-1)  # Attentionをsoftmaxで確率的な形に変換
        output = torch.matmul(normalized_weights, v)  # Attentionとvalueの掛け算
        output = self.out(output)

        return output, normalized_weights


class FeedForward(nn.Module):
    def __init__(self, model_dim, ff_dim=1024, dropout=0.1):
        super().__init__()

        self.linear_1 = nn.Linear(model_dim, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(ff_dim, model_dim)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.dropout(x)
        x = self.linear_2(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, model_dim, dropout=0.1):
        super().__init__()

        self.norm_1 = nn.LayerNorm(model_dim)
        self.norm_2 = nn.LayerNorm(model_dim)

        self.attention = Attention(model_dim)

        self.feadforward = FeedForward(model_dim)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x_normalized = self.norm_1(x)
        attention_weights, normalized_weights = self.attention(x_normalized, x_normalized, x_normalized, mask)

        attention_add_weights = x + self.dropout_1(attention_weights)

        x_normalized_2 = self.norm_2(attention_add_weights)
        x_normalized_2 = self.feadforward(x_normalized_2)
        output = attention_add_weights + self.dropout_2(x_normalized_2)

        return output, normalized_weights


class ClassificationHead(nn.Module):
    """ Transformer Blockの出力を用いて、クラス分類をする"""
    def __init__(self, model_dim=300, output_dim=2):
        super().__init__()

        self.linear = nn.Linear(model_dim, output_dim)

        nn.init.normal_(self.linear.weight, std=0.02)  # 重みの初期化
        nn.init.normal_(self.linear.bias, 0)  # バイアスの初期化

    def forward(self, x):
        x0 = x[:, 0, :]  # 各ミニバッチの各文章の先頭の単語の特徴量のみを取り出す
        out = self.linear(x0)

        return out


class TransformerClassification(nn.Module):
    def __init__(self, text_embedding_vectors, model_dim=300, max_seq_len=256, output_dim=2):
        super().__init__()

        self.net1 = Embedder(text_embedding_vectors)
        self.net2 = PositionalEncoder(model_dim=model_dim, max_seq_len=max_seq_len)
        self.net3_1 = TransformerBlock(model_dim=model_dim)
        self.net3_2 = TransformerBlock(model_dim=model_dim)
        self.net4 = ClassificationHead(output_dim=output_dim, model_dim=model_dim)

    def forward(self, x, mask):
        x1 = self.net1(x)
        x2 = self.net2(x1)
        x3_1, normalized_weights_1 = self.net3_1(x2, mask)
        x3_2, normalized_weights_2 = self.net3_2(x3_1, mask)
        x4 = self.net4(x3_2)

        return x4, normalized_weights_1, normalized_weights_2


def main():
    pass


if __name__ == '__main__':
    main()
