# coding=utf-8
from torch import nn


class Conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super(Conv2DBatchNormRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)  # 入力をメモリに保存せずにそのまま出力を計算。Trueで高速化

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        outputs = self.relu(x)

        return outputs


class Conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super(Conv2DBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = self.conv(x)
        outputs = self.batchnorm(x)

        return outputs


class FeatureMapConvolution(nn.Module):
    def __init__(self):
        super(FeatureMapConvolution, self).__init__()

        # conv1
        conv1_conf = [3, 64, 3, 2, 1, 1, False]
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias = conv1_conf
        self.cbnr1 = Conv2DBatchNormRelu(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)

        # conv2
        conv2_conf = [64, 64, 3, 1, 1, 1, False]
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias = conv2_conf
        self.cbnr2 = Conv2DBatchNormRelu(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)

        # conv3
        conv3_conf = [64, 128, 3, 1, 1, 1, False]
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias = conv3_conf
        self.cbnr3 = Conv2DBatchNormRelu(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.cbnr1(x)
        x = self.cbnr2(x)
        x = self.cbnr3(x)
        outputs = self.maxpool(x)

        return outputs


class BottleNeckPSP(nn.Module):
    """ residualだが、convを挟んでresidulaをするかしないかがIdentifyとの違い """
    def __init__(self, in_channels, mid_channels, out_channels, stride, dilation):
        super(BottleNeckPSP).__init__()

        self.cbr1 = Conv2DBatchNormRelu(in_channels, mid_channels, kernel_size=1, stride=1, padding=0,
                                        dilation=1, bias=False)
        self.cbr2 = Conv2DBatchNormRelu(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=dilation,
                                        dilation=dilation, bias=False)
        self.cb3 = Conv2DBatchNorm(mid_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                   dilation=1, bias=False)

        self.cb_residual = Conv2DBatchNorm(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                           dilation=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.cb3(self.cbr2(self.cbr1(x)))
        residual = self.cb_residual(x)
        return self.relu(conv + residual)


class BottleNeckIdentifyPSP(nn.Module):
    """ 入力と同じサイズの出力。 出力はresidualで特徴抽出したもの + 入力 """
    def __init__(self, in_channels, mid_channels, stride, dilation):
        super(BottleNeckIdentifyPSP).__init__()

        self.cbr1 = Conv2DBatchNormRelu(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                                        bias=False)
        self.cbr2 = Conv2DBatchNormRelu(mid_channels, mid_channels, kernel_size=3, stride=1, padding=dilation,
                                        dilation=dilation, bias=False)
        self.cb3 = Conv2DBatchNorm(mid_channels, in_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.cb3(self.cbr2(self.cbr1(x)))
        residual = x
        return self.relu(conv + residual)


class ResidualBlockPSP(nn.Sequential):
    # nn.Sequentialを拡張すると、__init__でadd_moduleするとその順にforwardされる。
    def __init__(self, n_blocks, in_channels, mid_channels, out_channels, stride, dilation):
        super(ResidualBlockPSP, self).__init__()

        self.add_module(
            name="block1",
            module=BottleNeckPSP(in_channels, mid_channels, out_channels, stride, dilation)
        )

        for i in range(n_blocks - 1):
            self.add_module(
                name="block" + str(i + 2),
                module=BottleNeckIdentifyPSP(out_channels, mid_channels, stride, dilation)
            )


class PSPNet(nn.Module):
    def __init__(self, n_classes):
        super(PSPNet, self).__init__()

        block_config = [3, 4, 6, 3]  # resnet50用
        img_size = 475
        img_size_8 = 60

        self.feature_conv = FeatureMapConvolution()

        self.feature_res1 = ResidualBlockPSP(n_blocks=block_config[0], in_channels=128, mid_channels=64,
                                             out_channels=256, stride=1, dilation=1)

        self.feature_res2 = ResidualBlockPSP(n_blocks=block_config[1], in_channels=256, mid_channels=128,
                                             out_channels=512, stride=2, dilation=1)

        self.feature_dilated_res1 = ResidualBlockPSP(n_blocks=block_config[2], in_channels=512, mid_channels=256,
                                                     out_channels=1024, stride=1, dilation=2)

        self.feature_dilated_res2 = ResidualBlockPSP(n_blocks=block_config[3], in_channels=1024, mid_channels=512,
                                                     out_channels=2048, stride=1, dilation=4)


if __name__ == '__main__':
    main()
