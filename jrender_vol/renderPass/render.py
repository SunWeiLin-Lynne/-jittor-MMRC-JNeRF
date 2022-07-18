import ipdb
import numpy as np
import jittor as jt
from jittor import nn
from ..camera import *
from ..rayMarching import *

def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
       chunk是并行处理的光束数量，ret是一个chunk（1024×32=32768）的结果，all_ret是一个batch的结果
    """
    all_ret = {}
    #ipdb.set_trace()# 查看rays_flat chunk
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        #ipdb.set_trace()# 查看ret
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
        if jt.flags.no_grad:
            jt.sync_all()
            
    #ipdb.set_trace()# 查看all_ret
    all_ret = {k : jt.concat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, focal, chunk=1024*32, rays=None, c2w=None, intrinsic=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = pinhole_get_rays(H, W, focal, c2w, intrinsic)
    else:
        # use provided ray batch
        # shape: rays[2,N_rand,3] rays_o[N_rand,3] rays_d[N_rand,3]
        rays_o, rays_d = rays
    #ipdb.set_trace()# 查看rays_o,rays_d

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            assert intrinsic is None
            rays_o, rays_d = pinhole_get_rays(H, W, focal, c2w_staticcam)
        viewdirs = viewdirs / jt.norm(viewdirs, p=2, dim=-1, keepdim=True)# 除以范数
        viewdirs = jt.reshape(viewdirs, [-1,3]).float()# [N_rand,3]

    sh = rays_d.shape # [N_rand, 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = jt.reshape(rays_o, [-1,3]).float()
    rays_d = jt.reshape(rays_d, [-1,3]).float()
    #ipdb.set_trace()# 查看rays_o,rays_d以及viewdirs

    near, far = near * jt.ones_like(rays_d[...,:1]), far * jt.ones_like(rays_d[...,:1])#near[N_rand,1],far[N_rand,1]
    #ipdb.set_trace()# 查看
    rays = jt.concat([rays_o, rays_d, near, far], -1)# [N_rand,8]{ray_o[N_rand,3],ray_d[N_rand,3],near[N_rand,1],far[N_rand,1]}
    #ipdb.set_trace()# 查看
    if use_viewdirs:
        rays = jt.concat([rays, viewdirs], -1)# [N_rand,11]{ray_o[N_rand,3],ray_d[N_rand,3],near[N_rand,1],far[N_rand,1],viewdirs[N_rand,3]}
    #ipdb.set_trace()# 查看

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    #ipdb.set_trace()# 查看
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = jt.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]
