# Jittor 可微渲染新视角生成赛题(JNeRF)

![主要结果](https://github.com/SunWeiLin-Lynne/jittor-MMRC-JNeRF/blob/main/img1.png)



## 简介
 本项目包含了第二届计图人工智能挑战赛赛题二-可微渲染新视角生成的代码实现。本项目的特点是采用NeRF的算法思想，使用计图(Jittor)框架的Jrender仓库对模型进行实现。

 本项目在五个场景数据集中训练对应的NeRF模型，并通过调整参数以及模型的训练方式，提高模型预测的准确率。目前JNeRF模型在A榜的测试数据中的PSNR均值之和达到了100.5843，在B榜的测试数据中已达到64.6952。

## 安装 

#### 运行环境
- ubuntu 18.04.5
- python >= 3.7
- jittor >= 1.3.0
- cuda >= 11.0

#### 安装依赖
使用JRender前需要安装好Jittor及其他安装包，如下：
- jittor
- imageio==2.9.0
- imageio-ffmpeg==0.4.3
- matplotlib==3.3.0
- configargparse==1.3
- tensorboard==1.14.0
- tqdm==4.46.0
- opencv-python==4.2.0.34

## 训练
安装好jittor和上述其他依赖包后可按照以下命令运行：
```
cd jrender
bash download_competition_data.sh
```
完成后，可以看到模型训练测试使用的场景数据集。本次比赛共包含5个测试场景，其中Easyship属于简单难度，Car、Coffee属于中等难度，Scar、Scarf属于高难度。训练模型时，针对一个场景数据集进行训练。以Easyship为例，可运行以下命令：
```
python demo7-nerf.py --config ./configs/Easyship.txt
```
同理，可以训练Car、Coffee、Scar、Scarf对应的模型。完成训练后，日志文件保存在./logs/场景名/expname/中。

## 推理
#### 测试
在测试集上的生成结果，以Easyship为例，可运行以下命令：
```
python test.py --config ./configs/Easyship.txt
```
其他场景同理。测试结果将保存至./test_result/中
#### 后处理
完成训练及测试后，对每个场景模型的部分结果进行后处理，最终结果将保存至./result中，可运行以下命令：
```
python findbdb.py --config ./configs/Post_config.txt
```

## 致谢

此项目基于论文*Jittor: a novel deep learning framework with meta-operators and unified graph execution*以及*NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis* 实现。

部分代码参考了 [jittor-baseline](https://github.com/Jittor/jrender)。
