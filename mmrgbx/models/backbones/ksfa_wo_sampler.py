from mmrgbx.registry import MODELS
from mmengine.model import BaseModule
from mmengine.model.weight_init import trunc_normal_
import torch
from mmpretrain.models.utils import resize_pos_embed
import torch.nn as nn


class KSAFusionModule(nn.Module):
    MAX_NUM_MODALITY = 100

    def __init__(
        self,
        dim,
        transformer: nn.TransformerDecoder,
        patch_size=(16, 16),
        topk=5,
    ):
        super(KSAFusionModule, self).__init__()
        self.trasnformer = transformer
        self.topk = topk
        emb_temp = nn.Embedding(self.MAX_NUM_MODALITY, dim, padding_idx=0)
        self.modality_query = nn.Parameter(
            emb_temp.weight.data.clone().view(1, self.MAX_NUM_MODALITY, dim),
            requires_grad=False,
        )

        self.patch_size = patch_size

    def forward(
        self,
        mainx,
        others: list,
        position_embedding=None,
        gt_segment=None,
        ps_segments=None,
    ):
        """
        mainx: [B, C, H, W]
        others: List[[B, C, H, W]]
        """
        B, C, H, W = mainx.shape

        if (H, W) != self.patch_size:
            resize_poseb = resize_pos_embed(
                position_embedding, self.patch_size, (H, W), num_extra_tokens=0
            )
        else:
            resize_poseb = position_embedding
        other_features = []
        for i, x in enumerate(others):
            assert x.shape == mainx.shape, (
                "the shape of mainx and others should be the same"
            )
            x = x.view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]
            query_token = self.modality_query[:, i + 1 : i + 2, :]  # [1, 1, C]
            x = x + query_token + resize_poseb
            other_features.append(x)

        main_query_token = self.modality_query[:, 0:1, :]  # [1, 1, C]
        main_feature = mainx.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, H*W, C]
        main_feature = main_feature + main_query_token + resize_poseb

        other_features = torch.cat(other_features, dim=1).contiguous()  # [B, 5*N, C]

        fusion_feature = self.trasnformer(main_feature, other_features)  # [B, H*W, C]
        fusion_feature = (
            fusion_feature.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        )  # [B, C, H, W]

        return fusion_feature, torch.zeros(
            1, device=fusion_feature.device, dtype=torch.float
        )


@MODELS.register_module()
class KSFAWOSamplerEncoder(BaseModule):
    def __init__(
        self,
        encoder_cfg: dict,
        init_cfg=None,
        in_channel=1024,
        embed_dim=768,
        patch_size=(16, 16),
        top_k=10,
        decoder_head=8,
        dropout=0.1,
        num_layer=3,
        frozen_stages=-1,
        *args,
        **kwargs,
    ):
        super(KSFAWOSamplerEncoder, self).__init__(init_cfg)

        self.encoder = MODELS.build(encoder_cfg)
        # layer_count = len(self.vit_encoder.layers)
        self.embed_dim = embed_dim
        num_patches = patch_size[0] * patch_size[1]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        self.project = nn.Conv2d(in_channel, embed_dim, 1)
        self.fusion_module = KSAFusionModule(
            dim=self.embed_dim,
            patch_size=patch_size,
            topk=top_k,
            transformer=nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=self.embed_dim,
                    nhead=decoder_head,
                    dim_feedforward=2048,
                    dropout=dropout,
                    batch_first=True,
                ),
                num_layers=num_layer,
            ),
        )
        self.frozen_stages = frozen_stages
        if self.frozen_stages > 0:
            self._freeze_stages()

    def init_weights(self):
        super(KSFAWOSamplerEncoder, self).init_weights()

        if not (
            isinstance(self.init_cfg, dict) and self.init_cfg["type"] == "Pretrained"
        ):
            if self.pos_embed is not None:
                trunc_normal_(self.pos_embed, std=0.02)

    def _freeze_stages(self):
        # freeze position embedding
        if self.pos_embed is not None:
            self.pos_embed.requires_grad = False

        self.encoder.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad = False

        # freeze layers
        for i in range(1, self.frozen_stages + 1):
            m = self.fusion_module.trasnformer.layers[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward_one(self, x):
        xs = self.encoder(x)
        if len(xs) == 1:
            return xs, self.project(xs[0])
        return xs[:-1], self.project(xs[-1])

    def pre_forward(self, xs):
        outs = []
        main_x = xs[0]
        others = xs[1:]
        main_x_pre, main_x = self.forward_one(main_x)
        with torch.no_grad():
            for x in others:
                _, x = self.forward_one(x)

                outs.append(x)
        return main_x_pre, main_x, outs

    def forward(self, x, gt_segment=None, ps_segments=None):
        main_x_pre, mainx, others = self.pre_forward(x)  # B C H W, {B C H W}
        outs = [*main_x_pre]
        fusion_feature, score_losss = self.fusion_module(
            mainx, others, self.pos_embed, gt_segment, ps_segments
        )
        outs.append(fusion_feature)
        outs.append(score_losss)
        return tuple(outs)

    def _format_output(self, x, hw):
        B = x.shape[0]
        return x.reshape(B, *hw, -1).permute(0, 3, 1, 2)
