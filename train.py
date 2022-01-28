import torch
from torch import nn
import torch.nn.functional as F
# from torch import autograd
from torch.utils.data import DataLoader,Subset
import numpy as np
import os
# from skatt_model import SKAtModel
from updownup_knn import UDUModel
from skatt_utils import LossModel, parse_arguments
from torch.utils.tensorboard import SummaryWriter
from data_load import DenoiseDataloader
from tqdm import tqdm
import open3d as o3d

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

def train(opt):
    data = DenoiseDataloader(opt.data_dir,opt.mean,opt.variance_max)
    train_dataset, test_dataset= data(n_pts=opt.num_points)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=False)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, pin_memory=False)
    index_subset = [20]
    viz_dataset = Subset(test_dataset, index_subset)
    viz_dataloader = DataLoader(viz_dataset, shuffle=False, pin_memory=False)
    if not os.path.exists(opt.summary_dir):
        os.makedirs(opt.summary_dir)
    if not os.path.exists(opt.model_dir):
        os.makedirs(opt.model_dir)
    print("Random Seed: ", opt.manualseed)
    np.random.seed(opt.manualseed)
    torch.manual_seed(opt.manualseed)
    
    objective = LossModel(use_rep=opt.repulsion,alpha=opt.alpha,knn=opt.k,max_rad=opt.max_rad)
    # model = SKAtModel(num_points=opt.num_points, num_heads=opt.num_heads, input_dim=opt.input_dim,cat_type=opt.cat_type).cuda()
    model = UDUModel(input_dim=opt.input_dim,num_points=opt.num_points,k=opt.k).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = opt.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, opt.milestones, gamma=0.1, last_epoch=-1)
    train_writer = SummaryWriter(opt.summary_dir)
    # optionally refine from a checkpoint
    if opt.refine:
        if os.path.isfile(opt.refine):
            print("=> loading checkpoint '{}'".format(opt.refine))
            checkpoint = torch.load(opt.refine)
            opt.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(opt.refine, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(opt.refine))

    n_batches = len(train_dataloader.dataset)
    test_size = len(test_dataloader.dataset)
    for epoch in range(opt.start_epoch,opt.nepochs):
        with tqdm(train_dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            model.train()
            for batch, (X, y) in enumerate(tepoch):
                X,y = X.float().to(device),y.float().to(device)
                pred = model(X)
                loss = objective.compute_loss(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print('[%d: %d/%d] train loss: %f\n' % (epoch, batch, n_batches, loss.item()))
                tepoch.set_postfix(loss=loss.item())
                train_writer.add_scalar('train_loss', loss.data.item(), epoch * n_batches + batch)
            
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for X, y in test_dataloader:
                    X,y = X.to(device),y.to(device)
                    pred = model(X)
                    test_loss += objective.chamfer_loss(pred, y).item()
            test_loss /= test_size
            train_writer.add_scalar('testloss', test_loss, epoch * n_batches + batch)
            print('validation loss: %f\n' % (test_loss))

            checpoint_state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()}
            if epoch == 0:
                with torch.no_grad():
                    for X, y in viz_dataloader:
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(np.squeeze(y.cpu().numpy()))
                        o3d.io.write_point_cloud("images/data_y.ply", pcd)
                        pcd.points = o3d.utility.Vector3dVector(np.squeeze(X.cpu().numpy()))
                        o3d.io.write_point_cloud("images/data_X.ply", pcd)
            if epoch == (opt.nepochs - 1):
                torch.save(checpoint_state, '%s/model_full_ae.pth' % opt.model_dir)
            if epoch % opt.modelsavefreq == 0:
                torch.save(checpoint_state, '%s/model_full_ae_%d.pth' % (opt.model_dir, epoch))
                with torch.no_grad():
                    for X, y in viz_dataloader:
                        X,y = X.to(device),y.to(device)
                        pred = model(X)
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(np.squeeze(pred.cpu().numpy()))
                        o3d.io.write_point_cloud("images/data_pred_%d.ply"%(epoch), pcd)

            scheduler.step()
    # train_writer.close()

if __name__ == '__main__':
    pars = parse_arguments()
    print(pars)
    train(pars)