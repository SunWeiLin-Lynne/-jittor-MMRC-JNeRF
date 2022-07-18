import ipdb
import os, sys
import numpy as np
import imageio
import json
import random
import time
import jittor as jt
from jittor import nn
from tqdm import tqdm, trange
import datetime
import matplotlib.pyplot as plt

from nerf_helper.utils import *
from nerf_helper.load_llff import load_llff_data
from nerf_helper.load_deepvoxels import load_dv_data
from nerf_helper.load_blender import load_blender_data

from tensorboardX import SummaryWriter
from jrender_vol.renderPass import render as render
from jrender_vol.camera import *
import csv #..
import codecs #..

jt.flags.use_cuda = 1
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        arr = []
        for i in range(0, inputs.shape[0], chunk):
            arr.append(fn(inputs[i:i+chunk]))
        return jt.concat(arr, 0)
    return ret

def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = jt.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = jt.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = jt.concat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)# netchunk通过网络并行发送的采样点数
    outputs = jt.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs
def render_path_val(i,img_i, H, W, focal, chunk, c2w, intrinsic, render_kwargs_test, target, savedir=None, expname=''):

    rgb, disp, acc, extras = render(H, W, focal, chunk=chunk, c2w=c2w, intrinsic=intrinsic,
                                    **render_kwargs_test)
    psnr = mse2psnr(img2mse(rgb, target))
    rgb = rgb.numpy()
    disp = disp.numpy()
    acc = acc.numpy()

    if savedir is not None:
        rgb_val8 = to8b(rgb)
        target_val8 = to8b(target)
        filename_r = os.path.join(savedir, expname + '_{:06d}_r_{:d}.png'.format(i, img_i))
        filename_t = os.path.join(savedir, expname + '_{:06d}_t_{:d}.png'.format(i, img_i))
        imageio.imwrite(filename_r, rgb_val8)
        imageio.imwrite(filename_t, target_val8)

    return rgb, disp,acc,extras,psnr
def render_path(render_poses, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0, intrinsic = None, expname=""):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(H, W, focal, chunk=chunk, c2w=c2w[:3,:4], intrinsic=intrinsic, **render_kwargs)
        rgbs.append(rgb.numpy())
        disps.append(disp.numpy())
        if i==0:
            print(rgb.shape, disp.shape)


        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, expname + '_r_{:d}.png'.format(i))
            imageio.imwrite(filename, rgb8)
        del rgb
        del disp
        del acc
        del _


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps

def find_bdb2D(image):
    # Args: image：800*800*3
    x_mean = image.mean(dim=0)# 按行求均值：求不同行对应rgb颜色的均值,shape:(行数,3),目的：找到图片中物体的位置
    y_mean = image.mean(dim=1)
    x_not_zero = []
    y_not_zero = []
    #ipdb.set_trace()
    white = jt.ones((1,3))
    #ipdb.set_trace()
    for i in range(len(x_mean)):
        #ipdb.set_trace()
        if jt.all(x_mean[i,:]<white):
            #ipdb.set_trace()
            x_not_zero.append(i)
    for j in range(len(y_mean)):
        #ipdb.set_trace()
        if jt.all(y_mean[j,:]<white):
            y_not_zero.append(j)
    #ipdb.set_trace()
    x1 = jt.min(x_not_zero)
    x2 = jt.max(x_not_zero)
    y1 = jt.min(y_not_zero)
    y2 = jt.max(y_not_zero)
    #ipdb.set_trace()
    return x1,x2,y1,y2

def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
    #ipdb.set_trace()#查看embed_fn位置编码的函数和input_ch：处理后的维度，是不是等于(3+args.multires?)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = jt.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer

def config_parser():
    gpu = "gpu"+os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/'+gpu+"/", 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=1e-3, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*8, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 
    parser.add_argument("--ubprecrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops in use batching')#................
    parser.add_argument("--object_detection", type=bool,
                        default=False, help='Whether to object_detection')  # ................


    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    parser.add_argument("--faketestskip", type=int, default=1, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')
    parser.add_argument("--near", type=float, default=2., 
                        help='set near distance')
    parser.add_argument("--far", type=float, default=12., 
                        help='set far distance')
    parser.add_argument("--do_intrinsic", action='store_true', 
                        help='use intrinsic matrix')
    parser.add_argument("--blender_factor", type=int, default=1, 
                        help='downsample factor for blender images')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=50000, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_tottest", type=int, default=400000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')

    return parser

def data_write_csv(base_dir,exp_name, datas, name):
    os.makedirs(os.path.join(base_dir, exp_name), exist_ok=True)
    datas_name = os.path.join(base_dir, exp_name, name)
    with open(datas_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(datas)

def train():

    parser = config_parser()
    args = parser.parse_args()

    # Load data
    intrinsic = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        testskip = args.testskip
        faketestskip = args.faketestskip
        if jt.mpi and jt.mpi.local_rank()!=0:
            testskip = faketestskip
            faketestskip = 1
        if args.do_intrinsic:
            images, poses, intrinsic, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip, args.blender_factor, True)
        else:
            images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip, args.blender_factor)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        #ipdb.set_trace()#查看下载的数据
        i_train, i_val, i_test = i_split
        i_test_tot = i_test
        i_test = i_test[::args.faketestskip]

        near = args.near
        far = args.far
        print(args.do_intrinsic)
        print("hwf", hwf)
        print("near", near)
        print("far", far)

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    render_poses = np.array(poses[i_test])
    #ipdb.set_trace()  # 查看下载的数据

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    """
     'network_query_fn':
     'perturb': 抖动
     'N_importance': 精采样数
     'network_fine': 精细网络
     'N_samples': 粗采样数
     'network_fn': 粗网络
     'use_viewdirs': 是否使用5D输入
     'white_bkgd'
     'raw_noise_std'
     'ndc': 仅适用于 LLFF 类型的数据，False
     'lindisp'
     'near': 积分的最近距离
     'far': 积分的最远距离
    """
    #ipdb.set_trace()#查看创建的NeRF

    # Move testing data to GPU
    render_poses = jt.array(render_poses)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with jt.no_grad():
            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    accumulation_steps = 1
    N_rand = args.N_rand//accumulation_steps
    use_batching = not args.no_batching
    
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, focal, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        #ipdb.set_trace()# 查看生成的光线
        print('done, concats')
        rays_c = np.stack([rays[i] for i in i_train], 0)
        images_c = np.stack([images[i] for i in i_train], 0)
        images_c = np.transpose(images_c[np.newaxis,:],[1,0,2,3,4])
        rays_rgb = np.concatenate([rays_c, images_c], 1) # [N, ro+rd+rgb, H, W, 3]
        #ipdb.set_trace()# 查看生成的光线
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        #ipdb.set_trace()# 查看变化
        #rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        #ipdb.set_trace()# 查看变化
        # 中心剪裁
        dH = int(H // 2 * args.ubprecrop_frac)
        dW = int(W // 2 * args.ubprecrop_frac)
        coords_ub = jt.stack(
            jt.meshgrid(
                jt.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                jt.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW)
            ), -1)  # (2dH,2dH,2)
        #ipdb.set_trace()
        coords_ub = jt.reshape(coords_ub, [-1,2])  # coords所有像素点的坐标(2dH * 2dW, 2)
        select_coords_ub = coords_ub[:].int()
        #ipdb.set_trace()
        rays_rgb = rays_rgb[:,select_coords_ub[:,0],select_coords_ub[:,1]]
        #ipdb.set_trace()# 查看变化
        
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)# 把生成的光线打乱
        #ipdb.set_trace()# 查看变化
        

        print('done')
        i_batch = 0

    # Move training data to GPU
    images = jt.array(images.astype(np.float32))
    poses = jt.array(poses)
    if use_batching:
        rays_rgb = jt.array(rays_rgb)


    N_iters = 51000 #..
    losses = []#..
    train_psnrs = []
    val_psnrs = []
    lrs = []
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    if not jt.mpi or jt.mpi.local_rank()==0:
        date = str(datetime.datetime.now())
        date = date[:date.rfind(":")].replace("-", "")\
                                        .replace(":", "")\
                                        .replace(" ", "_")
        gpu_idx = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        log_dir = os.path.join(basedir, "summaries", "log_" + date +"_gpu" + gpu_idx)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir=log_dir)

    start = start + 1
    for i in trange(start, N_iters):
        # jt.display_memory_info()
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            #ipdb.set_trace()# 查看变化
            batch = jt.transpose(batch, (1, 0, 2))
            #ipdb.set_trace()# 查看变化
            batch_rays, target_s = batch[:2], batch[2]
            #ipdb.set_trace()# 查看变化

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = jt.randperm(rays_rgb.shape[0])
                #ipdb.set_trace()# 查看变化
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image 随机选取一张图片
            np.random.seed(i)
            img_i = np.random.choice(i_train)
            print('img_i:',img_i)#.........................查看是否会选择相同的图片
            target = images[img_i]#.squeeze(0)
            pose = poses[img_i, :3,:4]#.squeeze(0)
            if N_rand is not None:
                # 生成该张图片的光线
                rays_o, rays_d = pinhole_get_rays(H, W, focal, pose, intrinsic)  # (H, W, 3), (H, W, 3)
                #ipdb.set_trace()# 查看rays_o,rays_d
                # 对前500iter的图像进行处理
                if i < args.precrop_iters:
                    #ipdb.set_trace()# 查看args.object_detection
                    if args.object_detection:# 对图像进行目标检测，检测后像素大小为（r2-r1+1，c2-c1+1）
                        c1, c2, r1, r2 = find_bdb2D(target)
                        #ipdb.set_trace()# 查看c1, c2, r1, r2,(r2-r1+1)*(c2-c1+1），可视化出来
                        if int((r2-r1+1)*(c2-c1+1)) < N_rand:
                            r1 = 0
                            c1 = 0
                            r2 = H-1
                            c2 = W-1
                        #ipdb.set_trace()
                        coords = jt.stack(
                            jt.meshgrid(
                                jt.linspace(int(r1), int(r2), int(r2-r1+1)),
                                jt.linspace(int(c1), int(c2), int(c2-c1+1))
                            ), -1)
                        #ipdb.set_trace()# 查看coords
                        if i == start:
                            print(f"[Config] Object detection of size {r2-r1+1} x {c2-c1+1}")
                    else:             # 对图像进行中心剪裁，剪裁后像素大小为（2dH,2dH）
                        dH = int(H//2 * args.precrop_frac)
                        dW = int(W//2 * args.precrop_frac)
                        coords = jt.stack(
                        jt.meshgrid(
                            jt.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            jt.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)# (2dH,2dH,2)
                        #ipdb.set_trace()# 查看dH,dW,coords,target
                        if i == start:
                            print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")
                else:
                    coords = jt.stack(jt.meshgrid(jt.linspace(0, H-1, H), jt.linspace(0, W-1, W)), -1)  # (H, W, 2)
                    #ipdb.set_trace()# 查看coords

                coords = jt.reshape(coords, [-1,2])  # (2dH * 2dW, 2)
                #ipdb.set_trace()# 查看coords
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # 从[0，coords.shape[0]）内选择N_rand个像素
                #ipdb.set_trace()# 查看select_inds
                select_coords = coords[select_inds].int()  # (N_rand, 2)选择的像素
                #ipdb.set_trace()# 查看
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)选择的像素对应的光线的原点
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)选择的像素对应的光线的方向向量
                batch_rays = jt.stack([rays_o, rays_d], 0)# (2，N_rand, 3) 选择的像素对应的光线 
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)选择的像素对应的原始图像
                #ipdb.set_trace()# 查看

        #####  Core optimization loop  #####
        rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)
        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][...,-1]
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)
        
        optimizer.backward(loss / accumulation_steps)
        if i % accumulation_steps == 0:
            optimizer.step()
        
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * accumulation_steps * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0

        # Rest is logging
        if (i+1)%args.i_weights==0 and (not jt.mpi or jt.mpi.local_rank()==0):
            print(i)
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            jt.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with jt.no_grad():
                rgbs, disps = render_path(render_poses, hwf, args.chunk, render_kwargs_test, intrinsic = intrinsic)
            if not jt.mpi or jt.mpi.local_rank()==0:
                print('Done, saving', rgbs.shape, disps.shape)
                moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
                print('movie base ', moviebase)
                imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
                imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

        
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
            losses.append(loss.item())
            train_psnrs.append(psnr.item())
            lrs.append(new_lrate)
            if i%args.i_img==0:
                img_i=np.random.choice(i_val)
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                valsavedir = os.path.join(basedir, expname, 'valset')
                os.makedirs(valsavedir, exist_ok=True)
                with jt.no_grad():
                    rgb, disp, acc, extras, val_psnr = render_path_val(i, img_i, H, W, focal, chunk=args.chunk, c2w=pose, intrinsic=intrinsic, render_kwargs_test=render_kwargs_test, target=target, savedir=valsavedir, expname=expname)
                    #rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose, intrinsic=intrinsic, **render_kwargs_test)
                    
#                 psnr = mse2psnr(img2mse(rgb, target))
#                 rgb = rgb.numpy()
#                 disp = disp.numpy()
#                 acc = acc.numpy()
                
                val_psnrs.append(val_psnr.item())

                if not jt.mpi or jt.mpi.local_rank()==0:
                    writer.add_image('test/rgb', to8b(rgb), global_step, dataformats="HWC")
                    writer.add_image('test/target', target.numpy(), global_step, dataformats="HWC")
                    writer.add_scalar('test/psnr', val_psnr.item(), global_step)
                
            jt.clean_graph()
            jt.sync_all()
            jt.gc()
        

            if i%args.i_testset==0 and i > 0:
                si_test = i_test_tot if i%args.i_tottest==0 else i_test
                testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
                os.makedirs(testsavedir, exist_ok=True)
                print('test poses shape', poses[si_test].shape)
                with jt.no_grad():
                    rgbs, disps = render_path(jt.array(poses[si_test]), hwf, args.chunk, render_kwargs_test, savedir=testsavedir, intrinsic = intrinsic, expname = expname)
                jt.gc()
        global_step += 1
    #data_write_csv("./logs/Easyship/summaries", iters, "/iters.csv")#..
    data_write_csv(basedir,expname, losses, "losses.csv")
    data_write_csv(basedir,expname, train_psnrs, "train_psnrs.csv")
    data_write_csv(basedir, expname, val_psnrs, "val_psnrs.csv")
    data_write_csv(basedir,expname, lrs, "lrs.csv")

if __name__=='__main__':
    train()
