import torch
import torch.nn as nn
from torch.nn import init
import torchvision
import functools
from copy import deepcopy
from mmrgbx.registry import MODELS

###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type="instance"):
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=True
        )
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def init_weights(net, init_type="normal", gain=0.02):
    net = net

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == "pretrained":
                pass
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None and init_type != "pretrained":
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    # print("initialize network with %s" % init_type)
    net.apply(init_func)


def init_net(net, init_type="normal", init_gain=0.02, gpu_ids=[]):
    for children in net.children():
        if children in net.need_initialization:
            init_weights(children, init_type, gain=init_gain)
        else:
            init_weights(children, "pretrained", gain=init_gain)  # for batchnorms
    return net


# def define_FuseNet(
#     num_labels,
#     rgb_enc=True,
#     depth_enc=True,
#     rgb_dec=True,
#     depth_dec=False,
#     norm="batch",
#     use_dropout=True,
#     init_type="xavier",
#     init_gain=0.02,
#     gpu_ids=[],
# ):
#     net = None
#     norm_layer = get_norm_layer(norm_type=norm)

#     net = FusenetModel(
#         num_labels, rgb_enc=True, depth_enc=True, rgb_dec=True, depth_dec=False
#     )
#     return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################


def VGG16_initializator():
    layer_names = [
        "conv1_1",
        "conv1_2",
        "conv2_1",
        "conv2_2",
        "conv3_1",
        "conv3_2",
        "conv3_3",
        "conv4_1",
        "conv4_2",
        "conv4_3",
        "conv5_1",
        "conv5_2",
        "conv5_3",
    ]
    layers = list(torchvision.models.vgg16(pretrained=True).features.children())
    layers = [x for x in layers if isinstance(x, nn.Conv2d)]
    layer_dic = dict(zip(layer_names, layers))
    return layer_dic


def make_layers_from_names(names, model_dic, bn_dim, existing_layer=None):
    layers = []
    if existing_layer is not None:
        layers = [
            existing_layer,
            nn.BatchNorm2d(bn_dim, momentum=0.1),
            nn.ReLU(inplace=True),
        ]
    for name in names:
        layers += [
            deepcopy(model_dic[name]),
            nn.BatchNorm2d(bn_dim, momentum=0.1),
            nn.ReLU(inplace=True),
        ]

    return nn.Sequential(*layers)


def make_layers_from_size(sizes):
    layers = []
    for size in sizes:
        layers += [
            nn.Conv2d(size[0], size[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(size[1], momentum=0.1),
            nn.ReLU(inplace=True),
        ]
    return nn.Sequential(*layers)


class MyUpsample2(nn.Module):
    def forward(self, x):
        return (
            x[:, :, :, None, :, None]
            .expand(-1, -1, -1, 2, -1, 2)
            .reshape(x.size(0), x.size(1), x.size(2) * 2, x.size(3) * 2)
        )


@MODELS.register_module()
class FusenetModel(nn.Module):
    def __init__(self):
        super(FusenetModel, self).__init__()
        self.need_initialization = []  # modules that need initialization
        model_dic = VGG16_initializator()

        ##### RGB ENCODER ####
        self.CBR1_RGB_ENC = make_layers_from_names(
            ["conv1_1", "conv1_2"], model_dic, 64
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.CBR2_RGB_ENC = make_layers_from_names(
            ["conv2_1", "conv2_2"], model_dic, 128
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.CBR3_RGB_ENC = make_layers_from_names(
            ["conv3_1", "conv3_2", "conv3_3"], model_dic, 256
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.dropout3 = nn.Dropout(p=0.4)

        self.CBR4_RGB_ENC = make_layers_from_names(
            ["conv4_1", "conv4_2", "conv4_3"], model_dic, 512
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.dropout4 = nn.Dropout(p=0.4)

        self.CBR5_RGB_ENC = make_layers_from_names(
            ["conv5_1", "conv5_2", "conv5_3"], model_dic, 512
        )
        self.dropout5 = nn.Dropout(p=0.4)

        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        feats_depth = list(
            torchvision.models.vgg16(pretrained=True).features.children()
        )
        # avg = torch.mean(feats_depth[0].weight.data, dim=1)
        # avg = avg.unsqueeze(1)

        conv11d = feats_depth[0]
        # conv11d.weight.data = feats_depth[0].weight.data

        self.CBR1_DEPTH_ENC = make_layers_from_names(
            ["conv1_2"], model_dic, 64, conv11d
        )
        self.pool1_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.CBR2_DEPTH_ENC = make_layers_from_names(
            ["conv2_1", "conv2_2"], model_dic, 128
        )
        self.pool2_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.CBR3_DEPTH_ENC = make_layers_from_names(
            ["conv3_1", "conv3_2", "conv3_3"], model_dic, 256
        )
        self.pool3_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.dropout3_d = nn.Dropout(p=0.4)

        self.CBR4_DEPTH_ENC = make_layers_from_names(
            ["conv4_1", "conv4_2", "conv4_3"], model_dic, 512
        )
        self.pool4_d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.dropout4_d = nn.Dropout(p=0.4)

        self.CBR5_DEPTH_ENC = make_layers_from_names(
            ["conv5_1", "conv5_2", "conv5_3"], model_dic, 512
        )

        ####  RGB DECODER  ####

        self.unpool5 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.CBR5_RGB_DEC = make_layers_from_size([[512, 512], [512, 512], [512, 512]])
        self.dropout5_dec = nn.Dropout(p=0.4)

        self.need_initialization.append(self.CBR5_RGB_DEC)

        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.CBR4_RGB_DEC = make_layers_from_size([[512, 512], [512, 512], [512, 256]])
        self.dropout4_dec = nn.Dropout(p=0.4)

        self.need_initialization.append(self.CBR4_RGB_DEC)

        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.CBR3_RGB_DEC = make_layers_from_size([[256, 256], [256, 256], [256, 128]])
        self.dropout3_dec = nn.Dropout(p=0.4)

        self.need_initialization.append(self.CBR3_RGB_DEC)

        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.CBR2_RGB_DEC = make_layers_from_size([[128, 128], [128, 64]])

        self.need_initialization.append(self.CBR2_RGB_DEC)

        init_net(self)

    def forward(self, x):
        rgb_inputs, depth_inputs = torch.split(x, 3, dim=1)[:2]
        ########  DEPTH ENCODER  ########
        # Stage 1
        # x = self.conv11d(depth_inputs)
        x_1 = self.CBR1_DEPTH_ENC(depth_inputs)
        x, id1_d = self.pool1_d(x_1)

        # Stage 2
        x_2 = self.CBR2_DEPTH_ENC(x)
        x, id2_d = self.pool2_d(x_2)

        # Stage 3
        x_3 = self.CBR3_DEPTH_ENC(x)
        x, id3_d = self.pool4_d(x_3)
        x = self.dropout3_d(x)

        # Stage 4
        x_4 = self.CBR4_DEPTH_ENC(x)
        x, id4_d = self.pool4_d(x_4)
        x = self.dropout4_d(x)

        # Stage 5
        x_5 = self.CBR5_DEPTH_ENC(x)

        ########  RGB ENCODER  ########

        # Stage 1
        y = self.CBR1_RGB_ENC(rgb_inputs)
        y = torch.add(y, x_1)
        y = torch.div(y, 2)
        y, id1 = self.pool1(y)

        # Stage 2
        y = self.CBR2_RGB_ENC(y)
        y = torch.add(y, x_2)
        y = torch.div(y, 2)
        y, id2 = self.pool2(y)

        # Stage 3
        y = self.CBR3_RGB_ENC(y)
        y = torch.add(y, x_3)
        y = torch.div(y, 2)
        y, id3 = self.pool3(y)
        y = self.dropout3(y)

        # Stage 4
        y = self.CBR4_RGB_ENC(y)
        y = torch.add(y, x_4)
        y = torch.div(y, 2)
        y, id4 = self.pool4(y)
        y = self.dropout4(y)

        # Stage 5
        y = self.CBR5_RGB_ENC(y)
        y = torch.add(y, x_5)
        y = torch.div(y, 2)
        y_size = y.size()

        y, id5 = self.pool5(y)
        y = self.dropout5(y)

        ########  DECODER  ########

        # Stage 5 dec
        y = self.unpool5(y, id5, output_size=y_size)
        y = self.CBR5_RGB_DEC(y)
        y1 = self.dropout5_dec(y)

        # Stage 4 dec
        y = self.unpool4(y1, id4)
        y = self.CBR4_RGB_DEC(y)
        y2 = self.dropout4_dec(y)

        # Stage 3 dec
        y = self.unpool3(y2, id3)
        y = self.CBR3_RGB_DEC(y)
        y3 = self.dropout3_dec(y)

        # Stage 2 dec
        y = self.unpool2(y3, id2)
        y4 = self.CBR2_RGB_DEC(y)

        return [y1, y2, y3, y4]
