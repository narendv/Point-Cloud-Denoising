from pc_transforms import fps
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import urllib
import ssl
import os
import zipfile
import glob
import h5py
import open3d as o3d

class DenoiseDataloader():
    def __init__(self,path, mean, variance_max, use_test=False):
        self.path = path
        self.use_test = use_test
        self.mean = mean
        # self.variance_train = variance_train
        # self.variance_test = variance_test
        self.variance_max = variance_max


    
    def load_data(self):
        train_pc = []
        val_pc = []
        test_pc = []
        train_list = glob.glob(os.path.join(self.path,'train/gt/*/*'))
        val_list = glob.glob(os.path.join(self.path,'val/gt/*/*'))

        if self.use_test:
            for f in val_list:
                with h5py.File(f) as m:
                    test_pc.append(torch.Tensor(np.array(m['data'])))
            print('Test dataset processed')
            
        else:
            for f in train_list:
                with h5py.File(f) as m:
                    train_pc.append(torch.Tensor(np.array(m['data'])))
            print('Training dataset compiled')

            for f in val_list:
                with h5py.File(f) as m:
                    val_pc.append(torch.Tensor(np.array(m['data'])))
            print('Val dataset compiled')
        
        return train_pc, val_pc, test_pc

    @staticmethod
    def down_sample(list, n_down):
        ds_list = []
        for pc in list:
            ds_list.append(fps(pc.unsqueeze(0).cuda(),n_down).squeeze(0))
        return ds_list

    @staticmethod
    def bounding_box(pc,var_type='rel'):
        if var_type == 'rel':
            x_min = torch.min(pc[:,0])
            x_max = torch.max(pc[:,0])
            y_min = torch.min(pc[:,1])
            y_max = torch.max(pc[:,1])
            z_min = torch.min(pc[:,2])
            z_max = torch.max(pc[:,2])
            bb_diag = torch.sqrt((x_max-x_min)**2 + (y_max-y_min)**2 + (z_max-z_min)**2 + 10**-8)
        elif var_type == 'abs':
            bb_diag = 1
        return bb_diag


    def gaussian_noise(self,pc_list,var_type='rel', train=True):
        torch.manual_seed(0)
        if train:
            clean_point_cloud = torch.cat([torch.stack(pc_list)]*len(self.variance_train),dim=0)
            pc_train = []
            for i in range(len(self.variance_train)):
                for j in range(len(pc_list)):
                    pc_list[j] = pc_list[j] + torch.randn_like(pc_list[j])*self.variance_train[i]*self.bounding_box(pc_list[j],var_type) + self.mean
                pc_train.append(torch.stack(pc_list))
            noise_point_cloud = torch.cat(pc_train,dim=0)
            

        else:
            clean_point_cloud = torch.cat([torch.stack(pc_list)]*len(self.variance_test),dim=0)
            pc_test = []
            for i in range(len(self.variance_test)):
                for j in range(len(pc_list)):
                    pc_list[j] = pc_list[j] + torch.randn_like(pc_list[j])*self.variance_test[i]*self.bounding_box(pc_list[j],var_type) + self.mean
                pc_test.append(torch.stack(pc_list))
            noise_point_cloud = torch.cat(pc_test,dim=0)

        print('Noise Added')
        return noise_point_cloud,clean_point_cloud

    def gaussian_noise_rand(self,pc_list):
        clean_point_cloud = torch.stack(pc_list)
        for j in range(len(pc_list)):
            pc_list[j] = pc_list[j] + torch.randn_like(pc_list[j])*random.uniform(0,self.variance_max) + self.mean
        noise_point_cloud = torch.stack(pc_list)

        print('Noise Added')
        return noise_point_cloud,clean_point_cloud

    def __call__(self,n_pts):
        if n_pts == 2048: 
            data_train, data_val, data_test = self.load_data()
        elif n_pts == 1024:
            data_train, data_val, data_test = self.load_data()
            data_train = self.down_sample(data_train,n_pts)
            data_val = self.down_sample(data_val,n_pts)
            data_test = self.down_sample(data_test,n_pts)

        if self.use_test:
            X_test,Y_test = self.gaussian_noise_rand(data_test)
            # X_test,Y_test = data_norm(X_test,Y_test)
            train_dataset = None
        else:
            X_train,Y_train = self.gaussian_noise_rand(data_train)
            # X_train,Y_train = data_norm(X_train,Y_train)
            X_test,Y_test = self.gaussian_noise_rand(data_val)
            # X_test,Y_test = data_norm(X_test,Y_test)
            train_dataset = TensorDataset(X_train,Y_train)

        test_dataset = TensorDataset(X_test,Y_test)
        print('Dataset ready')
        return train_dataset,test_dataset


def dataset(url,folder):
    filename = url.rpartition('/')[2].split('?')[0]
    path = os.path.join(folder, filename)
    # if os.path.exists(path):  # pragma: no cover
    #     if log:
    #         print('Using exist file', filename)
    #     return path

    # if log:
    #     print('Downloading', url)


    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, 'wb') as f:
        f.write(data.read())

    return path

def extract_zip(path, folder, log=True):
    if log:
        print('Extracting', path)
    with zipfile.ZipFile(path, 'r') as f:
        f.extractall(folder)

def data_norm(X,y):
    pc_norm_X = (X - torch.mean(X,1,keepdims=True))
    pc_norm_y = (y - torch.mean(y,1,keepdims=True))
    pc_norm_X /= torch.max(torch.linalg.norm(X,dim=-1,keepdims=True),1,keepdims=True)[0]
    pc_norm_y /= torch.max(torch.linalg.norm(y,dim=-1,keepdims=True),1,keepdims=True)[0]
    print('Data Normalized')
    return pc_norm_X,pc_norm_y
    
def PointNet_test(embed_dim,noise_percent,dataset=''):
    X = []
    y = []
    points = []
    mesh = o3d.geometry.TriangleMesh.create_sphere()
    pcd = mesh.sample_points_uniformly(number_of_points=embed_dim)
    y.append(torch.Tensor(np.asarray(pcd.points)))
    mesh = o3d.geometry.TriangleMesh.create_box()
    pcd = mesh.sample_points_uniformly(number_of_points=embed_dim)
    y.append(torch.Tensor(np.asarray(pcd.points)))
    mesh = o3d.geometry.TriangleMesh.create_cone()
    pcd = mesh.sample_points_uniformly(number_of_points=embed_dim)
    y.append(torch.Tensor(np.asarray(pcd.points)))
    mesh = o3d.geometry.TriangleMesh.create_arrow()
    pcd = mesh.sample_points_uniformly(number_of_points=embed_dim)
    y.append(torch.Tensor(np.asarray(pcd.points)))
    mesh = o3d.geometry.TriangleMesh.create_icosahedron()
    pcd = mesh.sample_points_uniformly(number_of_points=embed_dim)
    y.append(torch.Tensor(np.asarray(pcd.points)))
    mesh = o3d.geometry.TriangleMesh.create_octahedron()
    pcd = mesh.sample_points_uniformly(number_of_points=embed_dim)
    y.append(torch.Tensor(np.asarray(pcd.points)))
    mesh = o3d.geometry.TriangleMesh.create_tetrahedron()
    pcd = mesh.sample_points_uniformly(number_of_points=embed_dim)
    y.append(torch.Tensor(np.asarray(pcd.points)))
    # mesh = o3d.geometry.TriangleMesh.get_bunny_mesh()
    # pcd = mesh.sample_points_uniformly(number_of_points=embed_dim)
    # y.append(torch.Tensor(np.asarray(pcd.points)))

    y_test = torch.stack(y)
    bb = bounding_box(y)
    for i in range(y_test.size()[0]):
        pc = y_test[i] + torch.randn_like(y_test[i])*noise_percent*0.01*bb[i]
        X.append(pc)
    X_test = torch.stack(X)
    # X_test,y_test = data_norm(X_test,y_test)
    test_dataset = TensorDataset(X_test.transpose(-2,-1),y_test.transpose(-2,-1))
    test_dataloader = DataLoader(test_dataset, batch_size = 64, pin_memory=False)
    return test_dataloader,X,y

def data_pointfilter(bs):
    y = []
    X = []
    root_train = 'Pointfilter_data/Train/'
    root_test = 'Pointfilter_data/Test/'
    # Training data
    with open(os.path.join(root_train, 'train.txt')) as f:
        shape_names = f.readlines()
        shape_names = [x.strip() for x in shape_names]
        shape_names = list(filter(None, shape_names))
        for shape_ind, shape_name in enumerate(shape_names):
            if shape_ind % 6 == 0:
                temp = np.load(os.path.join(root_train, shape_name + '.npy'))
                y.extend([temp]*5)
            else:
                X.append(np.load(os.path.join(root_train, shape_name + '.npy')))
    X_train = torch.from_numpy(np.stack(X))
    y_train = torch.from_numpy(np.stack(y))

    # Test data
    y = []
    X = []

    with open(os.path.join(root_test, 'test.txt')) as f:
        shape_names = f.readlines()
        shape_names = [x.strip() for x in shape_names]
        shape_names = list(filter(None, shape_names))
        for shape_ind, shape_name in enumerate(shape_names):
            if shape_ind % 2 == 0:
                temp = np.load(os.path.join(root_test, shape_name + '.npy'))
                y.extend([temp])
            else:
                X.append(np.load(os.path.join(root_test, shape_name + '.npy')))
    X_test = torch.from_numpy(np.stack(X))
    y_test = torch.from_numpy(np.stack(y))

    train_dataset = TensorDataset(X_train.transpose(-2,-1),y_train.transpose(-2,-1))
    test_dataset = TensorDataset(X_test.transpose(-2,-1),y_test.transpose(-2,-1))
    train_dataloader = DataLoader(train_dataset, batch_size = bs, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size = bs, pin_memory=True)

    return train_dataloader,test_dataloader
