import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from mmrgbx.registry import MODELS


class Gated_Fusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        out = torch.cat([x, y], dim=1)
        G = self.gate(out)

        PG = x * G
        FG = y * (1 - G)

        return torch.cat([FG, PG], dim=1)


class Upsample(nn.Module):
    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True
        )
        return x


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, padding=1):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(
            nin, nin, kernel_size=kernel_size, padding=padding, groups=nin
        )
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class decoder_block(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(decoder_block, self).__init__()

        self.identity = nn.Sequential(
            Upsample(2, mode="bilinear"),
            nn.Conv2d(input_channels, output_channels, kernel_size=1, padding=0),
        )

        self.decode = nn.Sequential(
            Upsample(2, mode="bilinear"),
            nn.BatchNorm2d(input_channels),
            depthwise_separable_conv(input_channels, input_channels),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
            depthwise_separable_conv(input_channels, output_channels),
            nn.BatchNorm2d(output_channels),
        )

    def forward(self, x):
        residual = self.identity(x)

        out = self.decode(x)

        out += residual

        return out


@MODELS.register_module()
class CMGFNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()

        self.pretrained = pretrained

        resnet_features = torchvision.models.resnet34(pretrained=pretrained)

        self.enc_rgb1 = nn.Sequential(
            resnet_features.conv1,
            resnet_features.bn1,
            resnet_features.relu,
        )
        self.enc_rgb2 = nn.Sequential(resnet_features.maxpool, resnet_features.layer1)

        self.enc_rgb3 = resnet_features.layer2
        self.enc_rgb4 = resnet_features.layer3
        self.enc_rgb5 = resnet_features.layer4

        # DSM Encoder Part
        encoder_depth = torchvision.models.resnet34(pretrained=pretrained)

        self.enc_dsm1 = nn.Sequential(
            encoder_depth.conv1,
            encoder_depth.bn1,
            encoder_depth.relu,
        )
        self.enc_dsm2 = nn.Sequential(encoder_depth.maxpool, encoder_depth.layer1)

        self.enc_dsm3 = encoder_depth.layer2
        self.enc_dsm4 = encoder_depth.layer3
        self.enc_dsm5 = encoder_depth.layer4

        self.pool = nn.MaxPool2d(2)

        self.gate5 = Gated_Fusion(16)
        self.gate4 = Gated_Fusion(16)
        self.gate3 = Gated_Fusion(16)
        self.gate2 = Gated_Fusion(16)
        self.gate1 = Gated_Fusion(16)

        self.gate_final = Gated_Fusion(16)

        self.dconv6_rgb = decoder_block(16, 16)
        self.dconv5_rgb = decoder_block(16 + 16, 16)
        self.dconv4_rgb = decoder_block(16 + 16, 16)
        self.dconv3_rgb = decoder_block(16 + 16, 16)
        self.dconv2_rgb = decoder_block(16 + 16, 16)
        self.dconv1_rgb = decoder_block(16 + 16, 16)

        self.side6_rgb = nn.Conv2d(512, 16, kernel_size=1, padding=0)
        self.side5_rgb = nn.Conv2d(512, 16, kernel_size=1, padding=0)
        self.side4_rgb = nn.Conv2d(256, 16, kernel_size=1, padding=0)
        self.side3_rgb = nn.Conv2d(128, 16, kernel_size=1, padding=0)
        self.side2_rgb = nn.Conv2d(64, 16, kernel_size=1, padding=0)
        self.side1_rgb = nn.Conv2d(64, 16, kernel_size=1, padding=0)

        self.dconv6_cross = decoder_block(16, 16)
        self.dconv5_cross = decoder_block(16 + 16 + 16, 16)
        self.dconv4_cross = decoder_block(16 + 16 + 16, 16)
        self.dconv3_cross = decoder_block(16 + 16 + 16, 16)
        self.dconv2_cross = decoder_block(16 + 16 + 16, 16)
        self.dconv1_cross = decoder_block(16 + 16 + 16, 16)

        self.side6_cross = nn.Conv2d(512, 16, kernel_size=1, padding=0)
        self.side5_cross = nn.Conv2d(512, 16, kernel_size=1, padding=0)
        self.side4_cross = nn.Conv2d(256, 16, kernel_size=1, padding=0)
        self.side3_cross = nn.Conv2d(128, 16, kernel_size=1, padding=0)
        self.side2_cross = nn.Conv2d(64, 16, kernel_size=1, padding=0)
        self.side1_cross = nn.Conv2d(64, 16, kernel_size=1, padding=0)

    def forward(self, x):
        # dsm_encoder
        x_rgb, x_dsm = torch.split(x, 3, dim=1)
        y1 = self.enc_dsm1(x_dsm)  # bs * 64 * W/2 * H/2
        y1_side = self.side1_cross(y1)

        x1 = self.enc_rgb1(x_rgb)  # bs * 64 * W/2 * H/2
        x1_side = self.side1_rgb(x1)

        ##########################################################

        y2 = self.enc_dsm2(y1)  # bs * 64 * W/4 * H/4
        y2_side = self.side2_cross(y2)

        x2 = self.enc_rgb2(x1)  # bs * 64 * W/4 * H/4
        x2_side = self.side2_rgb(x2)

        ##########################################################

        y3 = self.enc_dsm3(y2)  # bs * 128 * W/8 * H/8
        y3_side = self.side3_cross(y3)

        x3 = self.enc_rgb3(x2)  # bs * 128 * W/8 * H/8
        x3_side = self.side3_rgb(x3)

        ##########################################################

        y4 = self.enc_dsm4(y3)  # bs * 256 * W/16 * H/16
        y4_side = self.side4_cross(y4)

        x4 = self.enc_rgb4(x3)  # bs * 256 * W/16 * H/16
        x4_side = self.side4_rgb(x4)

        ##########################################################

        y5 = self.enc_dsm5(y4)  # bs * 512 * W/16 * H/16
        y5_side = self.side5_cross(y5)

        x5 = self.enc_rgb5(x4)  # bs * 512 * W/16 * H/16
        x5_side = self.side5_rgb(x5)

        ##########################################################

        y6 = self.pool(y5)
        y6_side = self.side6_cross(y6)
        out_dsm1 = self.dconv6_cross(y6_side)

        x6 = self.pool(x5)
        x6_side = self.side6_rgb(x6)

        out_rgb1 = self.dconv6_rgb(x6_side)

        ##########################################################

        FG = torch.cat((x5_side, out_rgb1), dim=1)
        out_rgb2 = self.dconv5_rgb(FG)

        FG_cross = self.gate5(x5_side, y5_side)
        FG_dsm = torch.cat((FG_cross, out_dsm1), dim=1)
        out_dsm2 = self.dconv5_cross(FG_dsm)

        ##########################################################

        FG = torch.cat((x4_side, out_rgb2), dim=1)
        out_rgb3 = self.dconv4_rgb(FG)

        FG_cross = self.gate4(x4_side, y4_side)
        FG_dsm = torch.cat((FG_cross, out_dsm2), dim=1)
        out_dsm3 = self.dconv4_cross(FG_dsm)

        ##########################################################

        FG = torch.cat((x3_side, out_rgb3), dim=1)
        out_rgb4 = self.dconv3_rgb(FG)

        FG_cross = self.gate3(x3_side, y3_side)
        FG_dsm = torch.cat((FG_cross, out_dsm3), dim=1)
        out_dsm4 = self.dconv3_cross(FG_dsm)

        ##########################################################

        FG = torch.cat((x2_side, out_rgb4), dim=1)
        out_rgb5 = self.dconv2_rgb(FG)

        FG_cross = self.gate2(x2_side, y2_side)
        FG_dsm = torch.cat((FG_cross, out_dsm4), dim=1)
        out_dsm5 = self.dconv2_cross(FG_dsm)

        ##########################################################

        FG = torch.cat((x1_side, out_rgb5), dim=1)
        out_rgb6 = self.dconv1_rgb(FG)

        FG_cross = self.gate1(x1_side, y1_side)
        FG_dsm = torch.cat((FG_cross, out_dsm5), dim=1)
        out_dsm6 = self.dconv1_cross(FG_dsm)

        ##########################################################

        final_fused = self.gate_final(out_rgb6, out_dsm6)

        return [
            final_fused,
        ]
