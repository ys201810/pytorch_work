# coding=utf-8
from torch.nn import functional as F
from dataset import get_IMDb_Dataloader_and_text
from models import Embedder, PositionalEncoder, TransformerBlock, TransformerClassification


def main():
    train_dl, val_dl, test_dl, TEXT = get_IMDb_Dataloader_and_text(256, 24)

    # 動作確認
    batch = next(iter(train_dl))
    net1 = Embedder(TEXT.vocab.vectors)

    x = batch.Text[0]
    out1 = net1(x)
    print(x.shape)
    print(out1.shape)

    net2 = PositionalEncoder(model_dim=300, max_seq_len=256)
    out2 = net2(out1)
    print('入力のテンソルサイズ:{}'.format(out1.shape))
    print('出力のテンソルサイズ:{}'.format(out2.shape))

    net3 = TransformerBlock(model_dim=300)

    # maskの作成
    input_pad = 1  # <pad>のID=1
    input_mask = (x != input_pad)
    print(x)
    print(input_mask[0])

    out3, normalized_weight = net3(out2, input_mask)

    print("入力のテンソルサイズ:{}".format(out2.shape))
    print("出力のテンソルサイズ:{}".format(out3.shape))
    print("Attentionのサイズ:{}".format(normalized_weight.shape))

    net4 = TransformerClassification(text_embedding_vectors=TEXT.vocab.vectors, model_dim=300, max_seq_len=256,
                                     output_dim=2)

    out4, normalized_weight_1, normalized_weight_2 = net4(x, input_mask)
    print("出力のテンソルサイズ:{}".format(out4.shape))
    print("出力のテンソルのsigmoid:{}".format(F.softmax(out4, dim=1)))


if __name__ == '__main__':
    main()
