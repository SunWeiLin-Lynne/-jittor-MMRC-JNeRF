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
import csv
import codecs

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

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = jt.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    # Positional encoding
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
    #ipdb.set_trace()#查看embed_fn, imput_ch：63
    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
        #ipdb.set_trace()#查看embeddirs_fn, imput_ch_views：27
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    
    
    # Create model
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

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn, embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)
    #ipdb.set_trace()#查看model
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
    #ipdb.set_trace()#查看ckpts
    checkpoints = jt.load(ckpts[-1])
    #ipdb.set_trace()#查看checkpoints
    model.load_state_dict(checkpoints['network_fn_state_dict'])
    model_fine.load_state_dict(checkpoints['network_fine_state_dict'])
    #ipdb.set_trace()#查看model
    model.eval()
    model_fine.eval()
    #ipdb.set_trace()

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
    #ipdb.set_trace()#查看render_kwargs_test

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
    parser.add_argument("--testsavedir",type=str, default='./test_result',
                        help='The path of saving test result')

    return parser

def test():
    parser = config_parser()
    args = parser.parse_args()
    intrinsic = None
    #1) 加载测试集
    if args.dataset_type == 'blender':
        if args.do_intrinsic:  # 是否有相机的内参矩阵
            images, poses, intrinsic, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip,
                                                                                     args.blender_factor, 
                                                                                     True)
        else:
            images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip,
                                                                          args.blender_factor)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        #ipdb.set_trace()  # 查看下载的数据
        i_train, i_val, i_test = i_split # i_split : [[0:train], [train:val], [val:test]]
        #ipdb.set_trace()  #查看i_test

        near = args.near
        far = args.far
        print(args.do_intrinsic)
        print("hwf", hwf)
        print("near", near)
        print("far", far)

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    #2) 加载训练好的模型参数
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    
    global_step = start

    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    #ipdb.set_trace()  # 查看模型的创建结果

    #3) test
    #testsavedir = os.path.join(args.basedir, args.expname, 'testset')
    testsavedir = args.testsavedir
    os.makedirs(testsavedir, exist_ok=True)
    print('test poses shape', poses[i_test].shape)
    with jt.no_grad():
        rgbs, disps = render_path(jt.array(poses[i_test]), hwf, args.chunk, render_kwargs_test, savedir=testsavedir,
                                  intrinsic=intrinsic, expname=args.expname)
    jt.gc()

if __name__=='__main__':
    test()