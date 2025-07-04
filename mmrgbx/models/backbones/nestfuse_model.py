import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrgbx.registry import MODELS


EPSILON = 1e-5


# attention fusion strategy, average based on weight maps
def attention_fusion_weight(tensor1, tensor2, p_type):
    # avg, max, nuclear
    f_channel = channel_fusion(tensor1, tensor2, p_type)
    f_spatial = spatial_fusion(tensor1, tensor2)

    tensor_f = (f_channel + f_spatial) / 2
    return tensor_f


# select channel
def channel_fusion(tensor1, tensor2, p_type):
    # global max pooling
    shape = tensor1.size()
    # calculate channel attention
    global_p1 = channel_attention(tensor1, p_type)
    global_p2 = channel_attention(tensor2, p_type)

    # get weight map
    global_p_w1 = global_p1 / (global_p1 + global_p2 + EPSILON)
    global_p_w2 = global_p2 / (global_p1 + global_p2 + EPSILON)

    global_p_w1 = global_p_w1.repeat(1, 1, shape[2], shape[3])
    global_p_w2 = global_p_w2.repeat(1, 1, shape[2], shape[3])

    tensor_f = global_p_w1 * tensor1 + global_p_w2 * tensor2

    return tensor_f


def spatial_fusion(tensor1, tensor2, spatial_type="mean"):
    shape = tensor1.size()
    # calculate spatial attention
    spatial1 = spatial_attention(tensor1, spatial_type)
    spatial2 = spatial_attention(tensor2, spatial_type)

    # get weight map, soft-max
    spatial_w1 = torch.exp(spatial1) / (
        torch.exp(spatial1) + torch.exp(spatial2) + EPSILON
    )
    spatial_w2 = torch.exp(spatial2) / (
        torch.exp(spatial1) + torch.exp(spatial2) + EPSILON
    )

    spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
    spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)

    tensor_f = spatial_w1 * tensor1 + spatial_w2 * tensor2

    return tensor_f


# channel attention
def channel_attention(tensor, pooling_type="avg"):
    # global pooling
    shape = tensor.size()
    pooling_function = F.avg_pool2d

    if pooling_type == "avg":
        pooling_function = F.avg_pool2d
    elif pooling_type == "max":
        pooling_function = F.max_pool2d
    elif pooling_type == "nuclear":
        pooling_function = nuclear_pooling
    global_p = pooling_function(tensor, kernel_size=shape[2:])
    return global_p


# spatial attention
def spatial_attention(tensor, spatial_type="sum"):
    spatial = []
    if spatial_type == "mean":
        spatial = tensor.mean(dim=1, keepdim=True)
    elif spatial_type == "sum":
        spatial = tensor.sum(dim=1, keepdim=True)
    return spatial


# pooling function
def nuclear_pooling(tensor, kernel_size=None):
    shape = tensor.size()
    vectors = torch.zeros(1, shape[1], 1, 1).cuda()
    for i in range(shape[1]):
        u, s, v = torch.svd(tensor[0, i, :, :] + EPSILON)
        s_sum = torch.sum(s)
        vectors[0, i, 0, 0] = s_sum
    return vectors


class UpsampleReshape_eval(torch.nn.Module):
    def __init__(self):
        super(UpsampleReshape_eval, self).__init__()
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x1, x2):
        x2 = self.up(x2)
        shape_x1 = x1.size()
        shape_x2 = x2.size()
        left = 0
        right = 0
        top = 0
        bot = 0
        if shape_x1[3] != shape_x2[3]:
            lef_right = shape_x1[3] - shape_x2[3]
            if lef_right % 2 == 0.0:
                left = int(lef_right / 2)
                right = int(lef_right / 2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)

        if shape_x1[2] != shape_x2[2]:
            top_bot = shape_x1[2] - shape_x2[2]
            if top_bot % 2 == 0.0:
                top = int(top_bot / 2)
                bot = int(top_bot / 2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)

        reflection_padding = [left, right, top, bot]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x2 = reflection_pad(x2)
        return x2


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
            out = F.relu(out, inplace=True)
        return out


# light version
class DenseBlock_light(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseBlock_light, self).__init__()
        out_channels_def = int(in_channels / 2)
        # out_channels_def = out_channels
        denseblock = []

        denseblock += [
            ConvLayer(in_channels, out_channels_def, kernel_size, stride),
            ConvLayer(out_channels_def, out_channels, 1, stride),
        ]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out


@MODELS.register_module()
class NestFuseEncoder(nn.Module):
    def __init__(
        self,
        nb_filter=[64, 112, 160, 208, 256],
        input_nc=1,
    ):
        super(NestFuseEncoder, self).__init__()
        block = DenseBlock_light
        output_filter = 16
        kernel_size = 3
        stride = 1
        self.input_nc = input_nc

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2)
        self.up_eval = UpsampleReshape_eval()

        # encoder
        self.conv0 = ConvLayer(input_nc, output_filter, 1, stride)
        self.DB1_0 = block(output_filter, nb_filter[0], kernel_size, 1)
        self.DB2_0 = block(nb_filter[0], nb_filter[1], kernel_size, 1)
        self.DB3_0 = block(nb_filter[1], nb_filter[2], kernel_size, 1)
        self.DB4_0 = block(nb_filter[2], nb_filter[3], kernel_size, 1)
        # self.DB5_0 = block(nb_filter[3], nb_filter[4], kernel_size, 1)

        # decoder
        self.DB1_1 = block(nb_filter[0] + nb_filter[1], nb_filter[0], kernel_size, 1)
        self.DB2_1 = block(nb_filter[1] + nb_filter[2], nb_filter[1], kernel_size, 1)
        self.DB3_1 = block(nb_filter[2] + nb_filter[3], nb_filter[2], kernel_size, 1)

        self.DB1_2 = block(
            nb_filter[0] * 2 + nb_filter[1], nb_filter[0], kernel_size, 1
        )
        self.DB2_2 = block(
            nb_filter[1] * 2 + nb_filter[2], nb_filter[1], kernel_size, 1
        )

        self.DB1_3 = block(
            nb_filter[0] * 3 + nb_filter[1], nb_filter[0], kernel_size, 1
        )

    def encoder(self, input):
        x = self.conv0(input)
        x1_0 = self.DB1_0(x)
        x2_0 = self.DB2_0(self.pool(x1_0))
        x3_0 = self.DB3_0(self.pool(x2_0))
        x4_0 = self.DB4_0(self.pool(x3_0))
        # x5_0 = self.DB5_0(self.pool(x4_0))
        return [x1_0, x2_0, x3_0, x4_0]

    def forward(self, x):
        x1, x2 = torch.split(x, self.input_nc, dim=1)[:2]
        en1 = self.encoder(x1)
        en2 = self.encoder(x2)
        f_en = self.fusion(en1, en2, "max")
        de = self.decoder(f_en)
        return de

    def fusion(self, en1, en2, p_type):
        # attention weight
        fusion_function = attention_fusion_weight

        f1_0 = fusion_function(en1[0], en2[0], p_type)
        f2_0 = fusion_function(en1[1], en2[1], p_type)
        f3_0 = fusion_function(en1[2], en2[2], p_type)
        f4_0 = fusion_function(en1[3], en2[3], p_type)
        return [f1_0, f2_0, f3_0, f4_0]

    def decoder(self, f_en):
        x1_1 = self.DB1_1(torch.cat([f_en[0], self.up(f_en[1])], 1))

        x2_1 = self.DB2_1(torch.cat([f_en[1], self.up(f_en[2])], 1))
        x1_2 = self.DB1_2(torch.cat([f_en[0], x1_1, self.up(x2_1)], 1))

        x3_1 = self.DB3_1(torch.cat([f_en[2], self.up(f_en[3])], 1))
        x2_2 = self.DB2_2(torch.cat([f_en[1], x2_1, self.up(x3_1)], 1))

        x1_3 = self.DB1_3(torch.cat([f_en[0], x1_1, x1_2, self.up(x2_2)], 1))

        return [x1_1, x1_2, x1_3]


if __name__ == "__main__":
    input_nc = 1
    output_nc = 1
    deepsupervision = False
    nb_filter = [64, 112, 160, 208, 256]
    nest_model = NestFuseEncoder(nb_filter, input_nc, output_nc, deepsupervision)
    # input_size = [(2,256,256)]
    # print(summary(nest_model,input_size, device="cuda"))
    total_params = sum(p.numel() for p in nest_model.parameters() if p.requires_grad)
    print("Total parameters:", total_params)
