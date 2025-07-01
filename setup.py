#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="mmrgbx",  # 包的名称
    version="0.1.0",  # 包的版本
    author="Wang Tong",  # 包的作者
    author_email="kingcopper@whu.edu.cn",  # 作者的邮箱
    description="A brief description of the mmrgbx package",  # 包的简短描述
    url="https://github.com/yourusername/mmrgbx",  # 包的项目主页，通常是 GitHub 仓库地址
    packages=find_packages(
        exclude=("configs", "tools", "demo", ".vscode", "refcode")
    ),  # 自动查找包内的所有子包
    classifiers=[  # 项目的分类信息
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # 包所需的 Python 版本
    install_requires=[  # 安装包所需的依赖
        "numpy",
        "torch",
    ],
    # include_package_data=True,  # 是否包含包内的非代码文件，如 README、LICENSE 等
)
