# coding=utf-8
import math
import torch
from torch import nn
from attrdict import AttrDict
from torch.nn import functional as F

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

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)

        return self.gamma * x + self.beta


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        # 1. 単語IDをベクトル化
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)

        # 2. 位置情報テンソルをベクトル化(最大文字数の種類分のベクトル表現の作成)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # 3. 文章の1文目と2文目の情報をベクトル化
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        words_embeddings = self.word_embeddings(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        seq_length = input_ids.size(1)  # 文章の長さ
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # self attentionの特徴量を作成する全結合層
        self.query = nn.Linear(in_features=config.hidden_size, out_features=self.all_head_size)
        self.key = nn.Linear(in_features=config.hidden_size, out_features=self.all_head_size)
        self.value = nn.Linear(in_features=config.hidden_size, out_features=self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_status, attention_mask, attention_show_flg=False):
        """ hidden_status:EmbeddingsモジュールかBertLayerからの出力
            attention_mask:paddingを0埋めしたマスク
            attention_show_flg: Self Attentionの重みをreturnするかどうかのflg
        """
        mixed_query_layer = self.query(hidden_status)
        mixed_key_layer = self.key(hidden_status)
        mixed_value_layer = self.value(hidden_status)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # 特徴量同士を掛け算して似ている度合いをattention_scoreとして計算
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores + attention_mask  # attention_maskは-infか0で後のsoftmaxを考慮してadd

        # attentionの正規化
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        # attention mapを掛け算する
        context_layer = torch.matmul(attention_probs, value_layer)

        # multi-head Attentionのテンソルの形を元に戻す
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        if attention_show_flg:
            return context_layer, attention_probs
        else:
            return context_layer


class BertSelfOutput(nn.Module):
    """ BertSelfAttentionの出力を処理する全結合層 """
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_status, input_tensor):
        hidden_status = self.dense(hidden_status)
        hidden_status = self.dropout(hidden_status)
        hidden_status = self.LayerNorm(hidden_status + input_tensor)
        return hidden_status


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.selfattn = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, attention_show_flg=False):
        """ input_tensor: EmbeddingsかBertLayerからの出力 attention_mask:paddingを0埋めしたマスク """
        if attention_show_flg:
            self_output, attention_probs = self.selfattn(input_tensor, attention_mask, attention_show_flg)
            attention_output = self.output(self_output, input_tensor)
            return attention_output, attention_probs
        else:
            self_output = self.selfattn(input_tensor, attention_mask, attention_show_flg)
            attention_output = self.output(self_output, input_tensor)
            return attention_output


def gelu(x):
    """ Gaussian Error Linear Unit Leluの0以下で0になるところを滑らかにしたバージョン """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()

        self.dense = nn.Linear(in_features=config.hidden_size, out_features=config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_status):
        """ hidden_status: BertAttentionの出力 """
        hidden_status = self.dense(hidden_status)
        hidden_status = self.intermediate_act_fn(hidden_status)
        return hidden_status


class BertOutput(nn.Module):
    """ BERTの """
    def __init__(self, config):
        super(BertOutput, self).__init__()

        self.dense = nn.Linear(in_features=config.intermediate_size, out_features=config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()

        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_status, attention_mask, attention_show_flg=False):
        """
        :param hidden_status: Embedderモジュールの出力 [batch_size, seq_len, hidden_size]
        :param attention_mask: paddingを0埋めしたマスク
        :param attention_show_flg:
        :return:
        """
        if attention_show_flg:
            attention_output, attention_probs = self.attention(hidden_status, attention_mask, attention_show_flg)
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)
            return layer_output, attention_probs
        else:
            attention_output = self.attention(hidden_status, attention_mask, attention_show_flg)
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)
            return layer_output


class BertEncoder(nn.Module):
    """ BertLayerモジュールの繰り返し用 """
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_status, attention_mask, output_all_encoded_layers=True, attention_show_flg=False):
        """
        :param hidden_status: Embeddingsモジュールの出力
        :param attention_mask: paddingを0埋めしたマスク
        :param output_all_encoded_layers: 繰り返すBertLayerの各出力を返却するかどうかのflg
        :param attention_show_flg:
        :return:
        """
        all_encoder_layers = []

        for layer_module in self.layer:
            if attention_show_flg:
                hidden_status, attention_probs = layer_module(hidden_status, attention_mask, attention_show_flg)
            else:
                hidden_status = layer_module(hidden_status, attention_mask, attention_show_flg)

            if output_all_encoded_layers:  # returnに各BertLayerのoutputを返却するなら出力結果をつめる
                all_encoder_layers.append(hidden_status)

        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_status)

        if attention_show_flg:
            return all_encoder_layers, attention_probs
        else:
            return all_encoder_layers


class BertPooler(nn.Module):
    """ 入力文章の1単語目[cls]の特徴量を変換して保持するためのモジュール """
    def __init__(self, config):
        super(BertPooler, self).__init__()

        self.dense = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_status):
        # 1単語目の特徴量を取得
        first_token_tensor = hidden_status[:, 0]

        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(first_token_tensor)

        return pooled_output


class BertMoel(nn.Module):
    def __init__(self, config):
        super(BertMoel, self).__init__()

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True, attention_show_flg=False):
        """
        :param input_ids: [batch_size, sentence_length]の文章の単語IDの羅列
        :param token_type_ids: [batch_size, sequence_length]の各単語が1文目なのか、2文目なのかを示すID
        :param attention_mask: paddingが0埋めされたマスク
        :param output_all_encoded_layers: 最終の出力がBertLayerの各output(True)か最終だけ(False)かのflg
        :param attention_show_flg: Self-Attentionの出力を返すかのflg
        :return:
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # マスクの変形[batch_size, 1, 1, seq_length]
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)

        if attention_show_flg:
            encoded_layers, attention_probs = self.encoder(embedding_output, extended_attention_mask, output_all_encoded_layers, attention_show_flg)
        else:
            encoded_layers = self.encoder(embedding_output, extended_attention_mask, output_all_encoded_layers, attention_show_flg)

        pooled_output = self.pooler(encoded_layers[-1])

        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        if attention_show_flg:
            return encoded_layers, pooled_output, attention_probs
        else:
            return encoded_layers, pooled_output


def main():
    # 文章のIDデータの例の作成
    input_ids = torch.LongTensor([[31, 51, 12, 23, 99], [15, 5, 1, 0, 0]])
    print('入力の単語ID列のテンソルサイズ:{}'.format(input_ids.shape))

    # マスク作成
    attention_mask = torch.LongTensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    print('マスクのテンソルサイズ:{}'.format(attention_mask.shape))

    # 文章のIDの作成。0が1文目、1が2文目を示す。
    token_type_ids = torch.LongTensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])
    print('入力した文章IDのテンソルサイズ:{}'.format(token_type_ids.shape))

    net = BertMoel(config)

    encoded_layers, pooled_output, attention_probs = net(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, attention_show_flg=True)

    print('encoded_layersのテンソルサイズ:{}'.format(encoded_layers.shape))
    print('pooled_outputのテンソルサイズ:{}'.format(pooled_output.shape))
    print('attention_probs_layersのテンソルサイズ:{}'.format(attention_probs.shape))

    """
    embeddings = BertEmbeddings(config)
    encoder = BertEncoder(config)
    pooler = BertPooler(config)

    # maskのpadding箇所を-の大きい数字(softmaxで0が出力されるように)
    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    print('拡張したマスクのテンソルサイズ:{}'.format(extended_attention_mask.shape))

    # サンプルで処理
    out1 = embeddings(input_ids, token_type_ids)
    print('BertEmbeddingsの出力サイズ:{}'.format(out1.shape))

    out2 = encoder(out1, extended_attention_mask)
    print('BertEncoderの最終層の出力サイズ:{}'.format(out2[0].shape))

    out3 = pooler(out2[-1])  # output_all_encoded_layers=Trueなので最後の出力を利用
    print('BertEmbeddingsの出力サイズ:{}'.format(out3.shape))
    """

if __name__ == '__main__':
    main()
