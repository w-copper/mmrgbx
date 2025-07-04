import torch.nn as nn
import torch
import einops
from mmengine.model import BaseModel
from mmrgbx.registry import MODELS
from timm.models.layers import DropPath, trunc_normal_
import math


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        x = einops.rearrange(x, "b (h w) c k -> b c k h w", h=H, w=W)
        x = self.dwconv(x)
        x = einops.rearrange(x, "b c k h w -> b (h w) c k")

        return x


class MlpFuse(nn.Module):
    def __init__(
        self,
        input_k,
        input_dim,
        num_heads=8,
        downsample_rate: int = 1,
        hidden_dim=None,
        sr_ratio=1,
        drop_path=0.0,
    ) -> None:
        super().__init__()
        self.internal_dim = input_dim // downsample_rate
        hidden_dim = hidden_dim or self.internal_dim
        self.input_proj = nn.Linear(input_k, num_heads)
        self.mlp1 = Mlp(
            in_features=num_heads,
            hidden_features=hidden_dim,
            channels=input_dim,
            out_features=num_heads,
            act_layer=nn.GELU,
            drop=0.1,
        )
        self.mlp3 = Mlp(
            in_features=num_heads,
            hidden_features=hidden_dim,
            channels=input_dim,
            out_features=num_heads,
            act_layer=nn.GELU,
            drop=0.1,
        )

        self.norm1 = nn.LayerNorm([input_dim, num_heads])
        # self.norm2 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm([input_dim, num_heads])
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_atten = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor, H, W):
        """
        x shape(B, N_tokens, C, K_models)
        """

        short_cut = self.input_proj(x)
        x = self.norm1(short_cut)  # B Token C heads
        x = self.mlp1(x, H, W)
        out = short_cut + self.drop_path(self.mlp3(self.norm2(x), H, W))

        return out

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = (
                m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            )
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


class MlpFusion(nn.Module):
    def __init__(
        self,
        input_k,
        input_dim,
        output_k=8,
        downsample_rate: int = 1,
        hidden_dim=None,
        drop_path=0.0,
    ) -> None:
        super().__init__()
        self.internal_dim = input_dim // downsample_rate
        hidden_dim = hidden_dim or self.internal_dim
        self.input_proj = nn.Linear(input_k, output_k)
        self.mlp1 = Mlp(
            in_features=output_k,
            hidden_features=hidden_dim,
            channels=input_dim,
            out_features=output_k,
            act_layer=nn.GELU,
            drop=0.1,
        )
        self.mlp2 = Mlp(
            in_features=output_k,
            hidden_features=hidden_dim,
            channels=input_dim,
            out_features=output_k,
            act_layer=nn.GELU,
            drop=0.1,
        )

        self.norm1 = nn.LayerNorm([input_dim, output_k])
        self.norm2 = nn.LayerNorm([input_dim, output_k])
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor):
        """
        x shape(B, C, H, W , N)
        """
        _, _, H, W, _ = x.shape
        short_cut = self.input_proj(x)
        x = einops.rearrange(short_cut, "b c h w n -> b (h w) c n")
        # x =   # B Token C heads
        x = self.mlp1(self.norm1(x), H, W)
        out = self.drop_path(self.mlp2(self.norm2(x), H, W))
        out = einops.rearrange(out, "b (h w) c n -> b c h w n", h=H, w=W)
        out = short_cut + out

        return out

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = (
                m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            )
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


class AttentionFuse(nn.Module):
    def __init__(
        self,
        input_k,
        input_dim,
        num_heads=8,
        downsample_rate: int = 1,
        hidden_dim=None,
        sr_ratio=1,
        drop_path=0.0,
    ) -> None:
        super().__init__()
        self.internal_dim = input_dim // downsample_rate
        hidden_dim = hidden_dim or self.internal_dim
        self.input_proj = nn.Linear(input_k, num_heads)
        # self.mlp1 = Mlp(
        #     in_features=num_heads,
        #     hidden_features=hidden_dim,
        #     channels=input_dim,
        #     out_features=num_heads,
        #     act_layer=nn.GELU,
        #     drop=0.1,
        # )
        # self.mlp3 = Mlp(
        #     in_features=num_heads,
        #     hidden_features=hidden_dim,
        #     channels=input_dim,
        #     out_features=num_heads,
        #     act_layer=nn.GELU,
        #     drop=0.1,
        # )
        # self.mlp_out =  Mlp(in_features=input_k, hidden_features=hidden_dim, out_features=output_k, act_layer=nn.GELU, drop=0.1)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv3d(
                input_dim,
                input_dim,
                kernel_size=(1, sr_ratio, sr_ratio),
                stride=(1, sr_ratio, sr_ratio),
            )
            self.norm = nn.LayerNorm(input_dim)

        self.q_proj = nn.Linear(input_dim, self.internal_dim)
        self.k_proj = nn.Linear(input_dim, self.internal_dim)
        self.v_proj = nn.Linear(input_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, input_dim)

        self.norm1 = nn.LayerNorm([input_dim, num_heads])
        # self.norm2 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm([input_dim, num_heads])
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_atten = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor, H, W):
        """
        x shape(B, N_tokens, C, K_models)
        """

        short_cut = self.input_proj(x)
        x = self.norm1(short_cut)  # B Token C heads
        # x = self.mlp1(x, H, W)
        x = einops.rearrange(x, "b n c k -> b k n c")
        # x = x.permute(0, 3, 1, 2)
        q = self.q_proj(x)
        if self.sr_ratio > 1:
            # x = x.permute(0, 3, 1, 2)
            x = einops.rearrange(x, "b k (h w) c -> b c k h w", h=H, w=W)
            # x = x.reshape(B, C, n_heads, H, W)
            x = self.sr(x)
            x = einops.rearrange(x, "b c k h w -> b k (h w) c")
            x = self.norm(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        _, _, _, c_per_head = q.shape  # B heads Token C
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)
        out = attn @ v
        out = self.out_proj(out)
        out = self.drop_atten(out)

        out = einops.rearrange(out, "b k n c -> b n c k")
        out = short_cut + self.drop_path(self.norm2(out))

        return out

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = (
                m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            )
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


class MlpAttentionFuse(nn.Module):
    def __init__(
        self,
        input_k,
        input_dim,
        num_heads=8,
        downsample_rate: int = 1,
        hidden_dim=None,
        sr_ratio=1,
        drop_path=0.0,
    ) -> None:
        super().__init__()
        self.internal_dim = input_dim // downsample_rate
        hidden_dim = hidden_dim or self.internal_dim
        self.input_proj = nn.Linear(input_k, num_heads)
        self.mlp1 = Mlp(
            in_features=num_heads,
            hidden_features=hidden_dim,
            channels=input_dim,
            out_features=num_heads,
            act_layer=nn.GELU,
            drop=0.1,
        )
        self.mlp3 = Mlp(
            in_features=num_heads,
            hidden_features=hidden_dim,
            channels=input_dim,
            out_features=num_heads,
            act_layer=nn.GELU,
            drop=0.1,
        )
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv3d(
                input_dim,
                input_dim,
                kernel_size=(1, sr_ratio, sr_ratio),
                stride=(1, sr_ratio, sr_ratio),
            )
            self.norm = nn.LayerNorm(input_dim)

        self.q_proj = nn.Linear(input_dim, self.internal_dim)
        self.k_proj = nn.Linear(input_dim, self.internal_dim)
        self.v_proj = nn.Linear(input_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, input_dim)

        self.norm1 = nn.LayerNorm([input_dim, num_heads])
        # self.norm2 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm([input_dim, num_heads])
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_atten = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor, H, W, idx_to_save=None):
        """
        x shape(B, N_tokens, C, K_models)
        """
        if idx_to_save:
            torch.save(x, f"{idx_to_save}_origin")
        short_cut = self.input_proj(x)
        x = self.norm1(short_cut)  # B Token C heads
        x = self.mlp1(x, H, W)
        if idx_to_save:
            torch.save(x, f"{idx_to_save}_mlp1")
        x = einops.rearrange(x, "b n c k -> b k n c")
        # x = x.permute(0, 3, 1, 2)
        q = self.q_proj(x)
        if self.sr_ratio > 1:
            # x = x.permute(0, 3, 1, 2)
            x = einops.rearrange(x, "b k (h w) c -> b c k h w", h=H, w=W)
            # x = x.reshape(B, C, n_heads, H, W)
            x = self.sr(x)
            x = einops.rearrange(x, "b c k h w -> b k (h w) c")
            x = self.norm(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        _, _, _, c_per_head = q.shape  # B heads Token C
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)
        if idx_to_save:
            torch.save(attn, f"{idx_to_save}_attn")
        out = attn @ v
        out = self.out_proj(out)
        if idx_to_save:
            torch.save(attn, f"{idx_to_save}_af")
        out = self.drop_atten(out)

        out = einops.rearrange(out, "b k n c -> b n c k")
        out = short_cut + self.drop_path(self.mlp3(self.norm2(out), H, W))
        if idx_to_save:
            torch.save(out, f"{idx_to_save}_out")
        return out

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = (
                m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            )
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


@MODELS.register_module()
class LMFNet(BaseModel):
    def __init__(
        self,
        backbone,
        input_k,
        encoder_dims,
        fuse_srs=None,
        fuse_heads=None,
        stages=1,
        reduce="max",
        init_cfg=None,
    ):
        super(LMFNet, self).__init__(init_cfg)

        self.backbone = MODELS.build(backbone)
        self.encoder_dims = encoder_dims
        self.input_k = input_k
        self.reduce = reduce
        FuseModel = MlpAttentionFuse
        if stages == 1:
            self.fuse_model = nn.ModuleList(
                [
                    FuseModel(input_k, d, h, 2, sr_ratio=sr)
                    for d, h, sr in zip(encoder_dims, fuse_heads, fuse_srs)
                ]
            )
        elif stages > 1:
            self.fuse_model = nn.ModuleList(
                [
                    nn.ModuleList([FuseModel(input_k, d, h, 2, sr_ratio=sr)])
                    for d, h, sr in zip(encoder_dims, fuse_heads, fuse_srs)
                ]
            )
            for s in range(stages - 1):
                for i, m in enumerate(self.fuse_model):
                    m.append(
                        FuseModel(
                            fuse_heads[i],
                            encoder_dims[i],
                            num_heads=fuse_heads[i],
                            downsample_rate=i + 1,
                            sr_ratio=fuse_srs[i],
                        )
                    )
        self.stages = stages

        if self.reduce == "mlp":
            self.reduce_mlp = nn.ModuleList([nn.Linear(h, 1) for h in fuse_heads])

    def forward(self, x):
        batched_inputs = torch.split(x, 3, dim=1)[: self.input_k]

        features = [self.backbone(x) for x in batched_inputs]

        features_merged = []
        for i in range(len(self.encoder_dims)):
            x_i = [e[i] for e in features]
            x_i = torch.stack(x_i, dim=-1)
            b, c, h, w, s = x_i.shape
            x_i = einops.rearrange(x_i, "b c h w s -> b (h w) c s")
            if self.stages > 1:
                for m in self.fuse_model[i]:
                    x_i = m(x_i, h, w)
            else:
                x_i = self.fuse_model[i](x_i, h, w)
            if self.reduce == "mlp":
                m = self.reduce_mlp[i](x_i).squeeze(-1)
            elif self.reduce == "max":
                m = torch.max(x_i, dim=-1)[0]
            elif self.reduce == "avg":
                m = torch.mean(x_i, dim=-1)
            else:
                raise NotImplementedError()
            m = einops.rearrange(m, "b (h w) c -> b c h w", h=h, w=w)

            features_merged.append(m)

        return features_merged
