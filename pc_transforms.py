import numpy as np
from pointnet2_ops import pointnet2_utils
import torch

def shift_pc(batch_data, shift_range=0.1):
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        batch_data[batch_index,:,:] += shifts[batch_index,:]
    return batch_data

def random_scale_pc(batch_data, scale_low=0.8, scale_high=1.25):
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index,:,:] *= scales[batch_index]
    return batch_data

def compute_dist_mat(pc1, pc2):
    B, N, _ = pc1.shape
    _, M, _ = pc2.shape
    dist = -2 * torch.matmul(pc1, pc2.permute(0, 2, 1))
    dist += torch.sum(pc1 ** 2, -1).view(B, N, 1)
    dist += torch.sum(pc2 ** 2, -1).view(B, 1, M)
    return dist

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def index_pc(pc, indx):
    device = pc.device
    B = pc.shape[0]
    view_shape = list(indx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(indx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_pc = pc[batch_indices, indx, :]
    return new_pc

def fps(pc, n_pts):
    fps_indx = pointnet2_utils.furthest_point_sample(pc, n_pts).long()
    pc = index_pc(pc, fps_indx)
    return pc

def query_ball_point(nsample, xyz, new_xyz):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    sqrdists = compute_dist_mat(new_xyz, xyz)
    group_idx = torch.topk(sqrdists,nsample,-1,False)[1]
    return group_idx


def sample_and_group(npoint, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = pointnet2_utils.furthest_point_sample(xyz, npoint).long() # [B, npoint, C]
    new_xyz = index_pc(xyz, fps_idx)
    idx = query_ball_point(nsample, xyz, new_xyz)
    grouped_xyz = index_pc(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_pc(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points
