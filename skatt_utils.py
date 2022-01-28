# from pointnet2_ops import pointnet2_utils
import pc_transforms
from chamfer_distance import chamfer_distance
import torch
from torch import nn
import argparse


class LossModel:
    def __init__(self,use_rep=False,alpha=0.9,knn = 10, max_rad=1):
        self.use_rep = use_rep
        self.alpha = alpha
        self.knn = knn
        self.max_radius = max_rad
        self.chamfer_kernel = chamfer_distance.ChamferDistance()

    def chamfer_loss(self,y, gt):
        dist1, dist2 = self.chamfer_kernel(y,gt)
        loss = torch.mean(dist1) + torch.mean(dist2)
        return loss
        
    def repulsion_loss(self,y):
        n=y.shape[1]
        
        # KNN indices
        dist_mat = pc_transforms.compute_dist_mat(y, y)
        values = torch.topk(dist_mat, k=self.knn, largest=False, sorted=False)[0]

        m = nn.ReLU(inplace=True)
        net=torch.mul(torch.sqrt(values+10**-5),m(self.max_radius**2-values))
        loss = (torch.sum(net))/(n*self.knn)
        return loss

    def compute_loss(self,y,gt):
        loss = self.chamfer_loss(y,gt)
        if self.use_rep:
            dist_square = ((y - gt) ** 2).sum(2)
            loss = self.alpha*loss + (1-self.alpha)*torch.mean(torch.max(dist_square, 1)[0])
        return loss

def parse_arguments():
    parser = argparse.ArgumentParser()

    # naming / file handling
    parser.add_argument('--summary_dir', type=str, default='tensorboard_summary',
                        help='tensorboard summary folder')
    parser.add_argument('--data_dir', type=str, default='/home/catlab/ShapeCompletion/completion3d',
                        help='input folder (point clouds)')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='output folder (trained models)')
    parser.add_argument('--log_dir', type=str,
                        default='logs', help='training log folder')
    parser.add_argument('--viz_dir', type=str,
                        default='visualisation_summary', help='viz log folder')
    parser.add_argument('--modelsavefreq', type=int,
                        default='10', help='save model each n epochs')
    parser.add_argument('--refine', type=str, default='',
                        help='refine model at this path')

    # data parameters
    parser.add_argument('--mean', type=float, default=0.0,
                        help='mean of noisy data')
    parser.add_argument('--variance_max', type=list, default=0.05,
                        help='maximum variance of noisy data used for training')
    # model parameters
    parser.add_argument('--manualseed', type=int, default=171717,
                        help='random seed for numpy and pytorch')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='number of points sampled in a pc')
    parser.add_argument('--input_dim', type=int, default=3,
                        help='input channels of the pc')
    parser.add_argument('--alpha', type=int, default=0.60,
                        help='weight of chamfer loss in total loss')
    parser.add_argument('--k', type=int, default=32,
                        help='# of knn points considered')
    parser.add_argument('--max_rad', type=int, default=1,
                        help='points within this radius are considered for repulsion')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='# of attention heads used in Multihead Attention')
    parser.add_argument('--cat_type', type=str, default='res',
                        help='how to combine the enc-dec information')                                     
    parser.add_argument('--repulsion', type=bool, default=False,
                        help='whether to use repulsion or not')

    # training parameters
    parser.add_argument('--lr', type=int, default=1e-3,
                        help='learning rate')
    parser.add_argument('--milestones', type=list, default=[],
                        help='adjusting learning rate acc to epochs')
    parser.add_argument('--nepochs', type=int, default=100,
                        help='# of epochs')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='at which epoch to start training (might change when refining')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='learning rate')

    return parser.parse_args()
