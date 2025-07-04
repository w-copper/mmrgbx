from mmrgbx.registry import MODELS
from mmengine.model import BaseModule
from mmengine.model.weight_init import trunc_normal_

import torch
from mmpretrain.models.utils import resize_pos_embed
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class BlockwiseRandomSampler(nn.Module):
    def __init__(self, patch_size=16, k=2):
        """
        Block-wise 随机采样器
        :param patch_size: 块大小 P
        :param k: 每个块内采样点数 K
        """
        super(BlockwiseRandomSampler, self).__init__()
        self.patch_size = patch_size
        self.k = k

    def forward(self, x):
        """
        输入:
            x: 特征图 (B, ...)
        输出:
            coords: 候选点坐标 (B, N, 2) -> [x, y] in [-1, 1]
        """
        with torch.no_grad():
            device = x.device
            B = x.size(0)
            # 分块数量
            ph, pw = self.patch_size

            # 生成所有块的随机坐标
            block_size = 2.0 / ph
            block_coords = torch.rand(B, ph, pw, self.k, 2, device=device) * block_size
            # 计算每个块的起始位置

            h_starts, w_starts = torch.meshgrid(
                torch.linspace(-1, 1 - block_size, ph, device=device),
                torch.linspace(-1, 1 - block_size, pw, device=device),
            )
            h_starts = h_starts.view(1, ph, pw, 1)
            w_starts = w_starts.view(1, ph, pw, 1)
            block_coords[:, :, :, :, 0] += h_starts
            block_coords[:, :, :, :, 1] += w_starts

            coords = block_coords.view(B, ph * pw * self.k, 2)  # B N 2
            assert coords.min() >= -1 and coords.max() <= 1
            point_features = F.grid_sample(x, coords.unsqueeze(1))  # B C 1 N
            point_features = point_features.squeeze(2).transpose(1, 2)  # B N C

        return coords, point_features


class ConfidenceScorer(nn.Module):
    def __init__(self, embed_dim=128):
        """
        置信度打分模块：小MLP
        """
        super(ConfidenceScorer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, feats):
        """
        feats: 插值后的特征向量 (B, N, C)
        返回: 置信度分数 (B, N)
        """
        scores = self.mlp(feats).squeeze(-1)  # (B, N)
        return scores


class SoftAlignModule(nn.Module):
    def __init__(
        self,
        # dim,
        around=5,
        around_r=0.1,
    ):
        super(SoftAlignModule, self).__init__()
        self.around_r = around_r
        self.around = around

    def to_position_emb(self, points, pos_emb_2d):
        """
        points: [B, A, N, 2]
        """
        points_embd = F.grid_sample(pos_emb_2d, points)
        return points_embd  # B A N C

    def forward(self, x1, points_feature, points, pos_emb_2d):
        """
        x1: [B, C, H, W]
        x2: [B, C, H, W]
        points: [B, 2, N]
        """
        with torch.no_grad():
            points_grid = points.unsqueeze(1)  # B 1 N 2
            B, N, C = points_feature.shape
            rep_points_feature = points_feature.unsqueeze(1)  # B 1 N C
            rep_points_feature = rep_points_feature.expand(
                B, self.around, N, C
            ).permute(0, 3, 1, 2)  # B C A  N
            around = (
                (torch.rand(B, self.around, N, 2) * 2 - 0.5) * 2 * self.around_r
            )  # B A N 2
            around = around.to(points_feature.device)
            around_points_grid = points_grid + around  # B A N 2
            around_points_grid.clip_(-1, 1)
        around_points_emb = self.to_position_emb(
            around_points_grid, pos_emb_2d
        )  # B C A N

        around_points_feature = F.grid_sample(x1, around_points_grid)  # B C A N

        sim_score = F.cosine_similarity(
            rep_points_feature, around_points_feature, dim=1
        )  # B A N
        # print(sim_score.shape)
        weights = torch.softmax(sim_score, dim=1)  # B A N
        weights = weights.unsqueeze(1)  # B 1 A N

        weight_emb = around_points_emb * weights
        weight_emb = torch.sum(weight_emb, dim=2).permute(0, 2, 1)  # B N C

        points_feature = points_feature + weight_emb

        return points_feature


class ScoreLoss(nn.Module):
    def __init__(self, buff_size=10, ignore_index=None):
        super(ScoreLoss, self).__init__()
        self.buff_size = buff_size
        self.ignore_index = ignore_index

    def forward(self, scores, points, gt_segment, ps_segments):
        if gt_segment is None:
            return torch.tensor(0.0).type_as(scores), scores
        if ps_segments is None:
            return torch.tensor(0.0).type_as(scores), scores
        # print(ps_segments.shape)
        B, _, H, W = gt_segment.shape
        B, N, _ = points.shape

        with torch.no_grad():
            points = (points.detach() + 1) / 2
            points_x = points[:, :, 0] * W
            points_y = points[:, :, 1] * H
            points_x = points_x.type(torch.long)
            points_y = points_y.type(torch.long)
            buff_top_x = torch.clamp(points_x - self.buff_size, 0, W - 1)
            buff_top_y = torch.clamp(points_y - self.buff_size, 0, H - 1)
            buff_bottom_x = torch.clamp(points_x + self.buff_size, 0, W - 1)
            buff_bottom_y = torch.clamp(points_y + self.buff_size, 0, H - 1)
            scores_gt = []
            for b in range(B):
                score_b = []
                for n in range(N):
                    topx = buff_top_x[b, n]
                    topy = buff_top_y[b, n]
                    bottomx = buff_bottom_x[b, n]
                    bottomy = buff_bottom_y[b, n]
                    buff_gt = gt_segment[b, 0, topy:bottomy, topx:bottomx]
                    buff_ps = ps_segments[b, topy:bottomy, topx:bottomx]
                    scorebn = torch.sum(buff_gt == buff_ps) / (
                        (topx - bottomx) * (topy - bottomy)
                    )
                    scorebn = scorebn.abs()
                    score_b.append(scorebn)
                score_b = torch.stack(score_b, dim=0).reshape(-1)
                scores_gt.append(score_b)
            scores_gt = torch.stack(scores_gt, dim=0)
            scores_gt = torch.clip(scores_gt, 0, 1)

        loss = F.mse_loss(scores, scores_gt, reduction="mean")
        return loss, scores_gt


class KSAFusionModule(nn.Module):
    MAX_NUM_MODALITY = 100

    def __init__(
        self,
        dim,
        point_sampler: BlockwiseRandomSampler,
        score_module: ConfidenceScorer,
        soft_align_module: SoftAlignModule,
        transformer: nn.TransformerDecoder,
        score_loss: ScoreLoss,
        patch_size=(16, 16),
        topk=5,
    ):
        super(KSAFusionModule, self).__init__()
        self.trasnformer = transformer
        self.point_sampler = point_sampler
        self.score_module = score_module
        self.soft_align_module = soft_align_module
        self.score_loss = score_loss
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
        points_features = []
        score_losss = []
        # scores = []
        # points = []
        if (H, W) != self.patch_size:
            resize_poseb = resize_pos_embed(
                position_embedding, self.patch_size, (H, W), num_extra_tokens=0
            )
        else:
            resize_poseb = position_embedding

        poseb_2d = resize_poseb.view(1, H, W, -1).permute(0, 3, 1, 2).contiguous()
        poseb_2d_batch = poseb_2d.expand(B, -1, -1, -1)
        for i, x in enumerate(others):
            assert x.shape == mainx.shape, (
                "the shape of mainx and others should be the same"
            )
            point_x, point_feature = self.point_sampler(x)  # [B, N, 2], [B, N, C]
            score = self.score_module(point_feature)  # [B, N]
            if self.score_loss is not None:
                if ps_segments is not None:
                    loss, score = self.score_loss(
                        score, point_x, gt_segment, ps_segments[:, i]
                    )
                else:
                    loss = 0
                score_losss.append(loss)
            with torch.no_grad():
                topk_point = torch.topk(score, k=self.topk, dim=1)[1]  # [B, 5]
                point_x = point_x.gather(
                    1, topk_point.unsqueeze(-1).expand(-1, -1, 2)
                )  # [B, 5, 2]
                point_feature = point_feature.gather(
                    1, topk_point.unsqueeze(-1).expand(-1, -1, C)
                )  # [B, 5, C]

            align_point_feature = self.soft_align_module(
                mainx, point_feature, point_x, poseb_2d_batch
            )  # [B, 5, C]
            query_token = self.modality_query[:, i + 1 : i + 2, :]  # [1, 1, C]
            align_point_feature = align_point_feature + query_token
            points_features.append(align_point_feature)

        main_query_token = self.modality_query[:, 0:1, :]  # [1, 1, C]
        main_feature = mainx.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, H*W, C]
        main_feature = main_feature + main_query_token + resize_poseb

        points_features = torch.cat(points_features, dim=1).contiguous()  # [B, 5*N, C]

        fusion_feature = self.trasnformer(main_feature, points_features)  # [B, H*W, C]
        fusion_feature = (
            fusion_feature.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        )  # [B, C, H, W]

        score_losss = sum(score_losss) if len(score_losss) > 0 else 0

        return fusion_feature, score_losss


@MODELS.register_module()
class KSFAEncoder(BaseModule):
    def __init__(
        self,
        encoder_cfg: dict,
        init_cfg=None,
        in_channel=1024,
        embed_dim=None,
        patch_size=(16, 16),
        grid_size=(16, 16),
        sampler_k=2,
        align_around=5,
        align_r=0.5,
        score_buff=10,
        ignore_index=255,
        top_k=10,
        decoder_head=8,
        dropout=0.1,
        num_layer=3,
        frozen_stages=-1,
    ):
        super(KSFAEncoder, self).__init__(init_cfg)

        self.encoder = MODELS.build(encoder_cfg)
        # layer_count = len(self.vit_encoder.layers)
        if embed_dim is None:
            embed_dim = in_channel
        self.embed_dim = embed_dim
        num_patches = patch_size[0] * patch_size[1]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        self.project = nn.Conv2d(in_channel, embed_dim, 1)
        self.fusion_module = KSAFusionModule(
            dim=self.embed_dim,
            point_sampler=BlockwiseRandomSampler(grid_size, sampler_k),
            score_module=ConfidenceScorer(self.embed_dim),
            soft_align_module=SoftAlignModule(
                around=align_around,
                around_r=align_r,
            ),
            patch_size=patch_size,
            topk=top_k,
            score_loss=ScoreLoss(buff_size=score_buff, ignore_index=ignore_index),
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
        super(KSFAEncoder, self).init_weights()

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
