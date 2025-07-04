import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmrgbx.registry import MODELS


# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            # out = F.normalize(out)
            out = F.relu(out, inplace=True)
            # out = self.dropout(out)
        return out


# Dense convolution unit
class DenseConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseConv2d, self).__init__()
        self.dense_conv = ConvLayer(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.dense_conv(x)
        out = torch.cat([x, out], 1)
        return out


# Dense Block unit
class DenseBlock(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, stride):
        super(DenseBlock, self).__init__()
        out_channels_def = 16
        denseblock = []
        denseblock += [
            DenseConv2d(in_channels, out_channels_def, kernel_size, stride),
            DenseConv2d(
                in_channels + out_channels_def, out_channels_def, kernel_size, stride
            ),
            DenseConv2d(
                in_channels + out_channels_def * 2,
                out_channels_def,
                kernel_size,
                stride,
            ),
        ]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out


@MODELS.register_module()
class DenseFuseModel(nn.Module):
    def __init__(self, input_nc=1, output_nc=1):
        super(DenseFuseModel, self).__init__()
        denseblock = DenseBlock
        nb_filter = [16, 64, 32, 16]
        kernel_size = 3
        stride = 1
        self.input_nc = input_nc
        # encoder
        self.conv1 = ConvLayer(input_nc, nb_filter[0], 7, 2)
        self.DB1 = denseblock(nb_filter[0], kernel_size, stride)

        # decoder
        self.conv2 = ConvLayer(nb_filter[1], nb_filter[1], kernel_size, stride)
        self.conv3 = ConvLayer(nb_filter[1], nb_filter[2], kernel_size, stride)
        self.conv4 = ConvLayer(nb_filter[2], nb_filter[3], kernel_size, stride)
        self.conv5 = ConvLayer(nb_filter[3], output_nc, kernel_size, stride)

    def encoder(self, input):
        x1 = self.conv1(input)
        x_DB = self.DB1(x1)
        return [x_DB]

    def forward(self, x):
        x1, x2 = torch.split(x, self.input_nc, dim=1)[:2]
        en1 = self.encoder(x1)
        en2 = self.encoder(x2)
        en = self.fusion(en1, en2)
        de1 = self.decoder(en)
        return [de1]

    def fusion(self, en1, en2, strategy_type="addition"):
        f_0 = (en1[0] + en2[0]) / 2
        return [f_0]

    def decoder(self, f_en):
        x2 = self.conv2(f_en[0])
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        output = self.conv5(x4)

        return output


if __name__ == "__main__":
    input_nc = 1
    output_nc = 1
    densefuse_model = DenseFuseModel(input_nc, output_nc)
    # input_size = [(2,256,256)]
    # print(summary(nest_model,input_size, device="cuda"))
    total_params = sum(
        p.numel() for p in densefuse_model.parameters() if p.requires_grad
    )
    print("Total parameters:", total_params)
