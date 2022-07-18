本赛道的数据集，包含5个不同的场景，每个场景包含若干张图片及其对应的相机参数。 每个场景文件夹，包含train、val、test三个子文件夹，子文件夹中图片对应的相机参数分别存储在transfroms_train{val/test}.json文件中。

以transforms_train.json文件为例，其中"camera_angle_x"指的是相机参数弧度角。"file_path"对应每一帧图片的文件路径，"transform_matrix"对应该帧图片的相机参数。该相机参数是右手坐标系下，相机空间到世界空间的变换矩阵。

Baseline: https://github.com/Jittor/jrender（demo7-nerf.py）
参考文献： NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis
场景文件来源： https://benedikt-bitterli.me/resources/ https://www.mitsuba-renderer.org/download.html

1)在训练每个场景模型时，可运行以下命令，以Easyship为例：python demo7-nerf.py --config ./configs/Easyship.txt
2)在测试每个场景模型时，可运行以下命令，获得测试集的结果，保存至./test_result，以Easyship为例：python test.py --config ./configs/Easyship.txt
3)完成训练及测试后，对每个场景模型的部分结果进行后处理，最终结果保存至result中，可运行：python findbdb.py --config ./configs/Post_config.txt