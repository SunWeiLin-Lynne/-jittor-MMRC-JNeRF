import ipdb
import imageio
import os
import numpy as np
import jittor as jt
import json

def config_parser():
    gpu = "gpu" + os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--whites_json", type=str, default="{'Car_B_test_baseline_r_0.png': [1.,1.,1.],}",
                        help='divide object and background')
    parser.add_argument("--postprocess_imgs_path", type=str, default="./test_result/")
    parser.add_argument("--save_postprocess_imgs_path", type=str, default="./result/")
    return parser


def find_bdb2D(image, white):
    # Args: image：800*800*3
    x_mean = image.mean(0)  # 按行求均值：求不同行对应rgb颜色的均值,shape:(行数,3),目的：找到图片中物体的位置
    #ipdb.set_trace()
    y_mean = image.mean(1)
    #ipdb.set_trace()
    x_not_zero = []
    y_not_zero = []
    for i in range(len(x_mean)):
        #ipdb.set_trace()
        if all(x_mean[i, :] < white):
            # ipdb.set_trace()
            x_not_zero.append(i)
    for j in range(len(x_mean)):
        if all(y_mean[j, :] < white):
            y_not_zero.append(j)
    # ipdb.set_trace()
    x1 = jt.min(x_not_zero)
    x2 = jt.max(x_not_zero)
    y1 = jt.min(y_not_zero)
    y2 = jt.max(y_not_zero)
    return x1,x2,y1,y2



def denoising(c1,c2,r1,r2,image):
    #ipdb.set_trace()
    white = [255,255,255]
    for i in range(0,c1):
        for j in range(0,800):
            if all(image[j][i] != white):
                image[j][i] = white
    for i in range(c2+1,800):
        for j in range(0,800):
            if all(image[j][i] != white):
                image[j][i] = white
    for i in range(c1,c2):
        for j in range(0,r1):
            if all(image[j][i] != white):
                image[j][i] = white
    for i in range(c1,c2):
        for j in range(r2+1,800):
            if all(image[j][i] != white):
                image[j][i] = white
    return image


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    #ipdb.set_trace()
    whites = json.loads(args.whites_json)
    #1) 加载test_result的所有图片
    images_path = os.listdir(args.postprocess_imgs_path)
    #ipdb.set_trace()
    #2) 针对每张图片根据white的不同进行处理
    for i in images_path:
        #ipdb.set_trace()
        image = imageio.imread(args.postprocess_imgs_path+i)
        #ipdb.set_trace()
        if whites[i]<[1.0,1.0,1.0]: # 需要后处理
            #ipdb.set_trace()
            image1 = image[..., :3] / 255
            c1, c2, r1, r2 = find_bdb2D(image1,whites[i])
            c1 = int(c1)
            c2 = int(c2)
            r1 = int(r1)
            r2 = int(r2)
            #ipdb.set_trace()
            image = denoising(c1, c2, r1, r2, image)
        #3) 修改名字并保存
        name_i_list = i.split("_")
        name_i = name_i_list[0]+"_"+name_i_list[-2]+"_"+name_i_list[-1]
        #ipdb.set_trace()
        imageio.imwrite(args.save_postprocess_imgs_path+name_i, image)