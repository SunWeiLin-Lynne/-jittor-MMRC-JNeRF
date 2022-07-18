# Jittor 可微渲染新视角生成赛题(JNeRF)
| 标题名称包含赛题、方法

![主要结果](https://s3.bmp.ovh/imgs/2022/04/19/440f015864695c92.png)

｜展示方法的流程特点或者主要结果等

## 简介
| 简单介绍项目背景、项目特点
本项目包含了第二届计图人工智能挑战赛赛题二-可微渲染新视角生成的代码实现。本项目采用NeRF算法的思路，并使用计图(Jittor)框架的Jrender仓库实现。通过调整参数以及模型的训练方式，在五个场景数据集中训练出对应的NeRF模型。目前JNeRF模型在A榜的测试数据中的PSNR达到了100.5843，在B榜的测试数据中已达到64.6952。

## 安装 
| 介绍基本的硬件需求、运行环境、依赖安装方法

本项目可在 2 张 2080 上运行，训练时间约为 6 小时。

#### 运行环境
- ubuntu 18.04.5
- python >= 3.7
- jittor >= 1.3.0
- cuda >= 11.0

#### 安装依赖
使用JRender前需要安装好Jittor及其他安装包，如下：
jittor
imageio==2.9.0
imageio-ffmpeg==0.4.3
matplotlib==3.3.0
configargparse==1.3
tensorboard==1.14.0
tqdm==4.46.0
opencv-python==4.2.0.34

## 训练
｜ 介绍模型训练的方法

单卡训练可运行以下命令：
```
bash scripts/train.sh
```

多卡训练可以运行以下命令：
```
bash scripts/train-multigpu.sh
```

## 推理
｜ 介绍模型推理、测试、或者评估的方法

生成测试集上的结果可以运行以下命令：

```
bash scripts/test.sh
```

## 致谢
| 对参考的论文、开源库予以致谢，可选

此项目基于论文 *A Style-Based Generator Architecture for Generative Adversarial Networks* 实现，部分代码参考了 [jittor-gan](https://github.com/Jittor/gan-jittor)。

## 注意事项

点击项目的“设置”，在Description一栏中添加项目描述，需要包含“jittor”字样。同时在Topics中需要添加jittor。

![image-20220419164035639](https://s3.bmp.ovh/imgs/2022/04/19/6a3aa627eab5f159.png)
