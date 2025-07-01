# MMRGBX

一个基于MMSegmentation的多模态遥感图像分割框架，专注于RGB和其他模态（如SAR、LiDAR等）数据的融合分割任务。

## 🌟 特性

- **多模态融合**: 支持RGB与多种模态数据（SAR、LiDAR、DSM等）的融合分割
- **丰富的模型**: 集成了多种先进的融合网络架构
- **多数据集支持**: 支持Potsdam、DFC23、C2Seg、OptSAR等多个遥感数据集
- **灵活配置**: 基于MMEngine的配置系统，易于扩展和定制
- **高效训练**: 支持分布式训练和混合精度训练

## 📦 安装

### 环境要求

- Python >= 3.6
- PyTorch >= 1.8
- CUDA (推荐)

### 依赖安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/mmrgbx.git
cd mmrgbx

# 安装依赖
pip install -r requirements.txt

# 安装mmrgbx
pip install -e .
```

### MMSegmentation生态系统依赖

本项目依赖以下版本的MMSegmentation生态系统：

- mmcv: >=2.0.0rc4, <2.3.0
- mmengine: >=0.6.0, <1.0.0
- mmseg: >=1.0.0rc6, <1.3.0

## 🚀 快速开始

### 数据准备

请参考 `tools/datasets/` 目录下的脚本来准备相应的数据集：

- Potsdam: `tools/datasets/potsdam.py`
- DFC23: `tools/datasets/dfc23track1.py`
- C2Seg: `tools/datasets/c2seg.py`

### 训练

```bash
# 单GPU训练
python tools/train.py configs/cmfg/cmfg-potsdam.py

# 多GPU训练
python -m torch.distributed.launch --nproc_per_node=4 tools/train.py configs/cmfg/cmfg-potsdam.py --launcher pytorch

# 使用SLURM
bash tools/slurm_train_4g16c.sh
```

### 测试

```bash
# 测试模型
python tools/test.py configs/cmfg/cmfg-potsdam.py checkpoints/model.pth

# 测试并保存结果
python tools/test.py configs/cmfg/cmfg-potsdam.py checkpoints/model.pth --out results/
```

## 🏗️ 模型架构

### 支持的融合网络

- **CMFG**: Cross-Modal Feature Guidance Network
- **CMX**: Cross-Modal eXchange Network
  - CMX-Segformer: 基于Segformer的跨模态交换网络
  - CMX-Swin: 基于Swin Transformer的跨模态交换网络
- **DATFuse**: Dual Attention Transformer Fusion
- **DenseFuse**: Dense Feature Fusion Network
- **FTransUnet**: Fusion Transformer U-Net
  - FTransUnet-ViTB: 基于Vision Transformer Base的融合网络
  - FTransUnet-ViTB-ResNet50: 结合ResNet50的融合网络
- **FuseNet**: Early Fusion Network
- **NestFusion**: Nested Fusion Network
- **Simple Concat**: 简单拼接融合基线
- **Dual Networks**: 双分支网络架构
  - Dual-ConvNeXt: 双分支ConvNeXt网络
  - Dual-MobileNetV2: 双分支MobileNetV2网络
  - Dual-EMO: 双分支EMO网络
  - Dual-SMT: 双分支SMT网络
  - Dual-UniFormer: 双分支UniFormer网络

### 单模态基准模型

- **SegFormer-B0**: 轻量级语义分割模型
- **UPerNet-ResNet50**: 基于ResNet50的UPerNet
- **UPerNet-Swin**: 基于Swin Transformer的UPerNet
- **PCPVT**: Pyramid Convolution and Vision Transformer


### 支持的数据集

- **Potsdam**: ISPRS Potsdam多模态数据集
- **DFC23**: IEEE GRSS DFC 2023 Track 1
- **C2Seg**: 变化检测分割数据集
- **OptSAR**: 光学-SAR融合数据集

## 📁 项目结构

```
mmrgbx/
├── configs/                 # 配置文件
│   ├── _base_/             # 基础配置
│   ├── cmfg/               # CMFG模型配置
│   ├── cmx-segformer/      # CMX-Segformer配置
│   ├── cmx-swin/           # CMX-Swin配置
│   └── ...
├── mmrgbx/                 # 核心代码
│   ├── datasets/           # 数据集定义
│   ├── models/             # 模型定义
│   │   ├── backbones/      # 骨干网络
│   │   ├── necks/          # 特征融合模块
│   │   └── segmentor/      # 分割器
│   └── optim/              # 优化器
└── tools/                  # 工具脚本
    ├── train.py            # 训练脚本
    ├── test.py             # 测试脚本
    └── datasets/           # 数据集处理脚本
```

## 🔧 配置系统

本项目使用MMEngine的配置系统，支持继承和组合：

```python
# 示例配置文件
_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/cmfg.py',
    '../_base_/datasets/potsdam.py',
    '../_base_/schedule.py',
]

model = dict(
    decode_head=dict(
        num_classes=6,
        loss_decode=[
            dict(type='mmseg.CrossEntropyLoss', use_sigmoid=False),
            dict(type='mmseg.DiceLoss', loss_weight=0.4),
        ],
    ),
)
```

## TODO:

- [x] 完善文档
- [x] 支持更多数据集
- [x] 支持更多融合网络
- [ ] 数据集预处理代码

## 🤝 贡献

欢迎贡献代码！请遵循以下步骤：

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本项目基于Apache 2.0许可证开源。详见 [LICENSE](LICENSE) 文件。

## 📞 联系方式

- 作者: Wang Tong
- 邮箱: kingcopper@whu.edu.cn
- 项目链接: [https://github.com/yourusername/mmrgbx](https://github.com/w-copper/mmrgbx)

## 🙏 致谢

本项目基于以下优秀的开源项目：

- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
- [MMEngine](https://github.com/open-mmlab/mmengine)
- [MMCV](https://github.com/open-mmlab/mmcv)

特别感谢 [Claude](https://claude.ai) 在项目文档编写和代码优化方面提供的智能辅助。

## 📚 引用

如果本项目对您的研究有帮助，请考虑引用：

```bibtex
@misc{mmrgbx2024,
  title={MMRGBX: A Multi-Modal Remote Sensing Image Segmentation Framework},
  author={Wang, Tong},
  year={2024},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={\url{https://github.com/yourusername/mmrgbx}}
}
```
