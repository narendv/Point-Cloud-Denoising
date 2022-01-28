from cmath import tanh
import bleach
import torch
from torch import nn
from torch.nn import MultiheadAttention
import torch.nn.functional as F
from torchsummary import summary
from pc_transforms import fps,index_pc,sample_and_group

class PCdown(nn.Module):
    def __init__(self, input_dim=3, output_dim=64, k = 5, type = 'enc2', last=False):
        super(PCdown, self).__init__()
        self.input_dim = input_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k = k
        self.mid_dim = output_dim*2 if type == 'enc2' else output_dim//2


        # self.conv1 = nn.Conv1d(input_dim, self.mid_dim, kernel_size=1,padding=0)
        # self.conv2 = nn.Conv1d(self.mid_dim, output_dim, kernel_size=1, padding=0)
        # self.conv3 = nn.Conv1d(output_dim+input_dim, output_dim, kernel_size=1,padding=0)
        # self.conv4 = nn.Conv1d(output_dim, output_dim, kernel_size=1, padding=0)

        self.first_mlp = nn.Sequential(
            nn.Conv1d(input_dim, self.mid_dim,1),
            nn.BatchNorm1d(self.mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.mid_dim, output_dim, 1)
        )
        self.last_mlp = nn.Sequential(
            nn.Conv1d(output_dim+input_dim, output_dim, 1),
            nn.BatchNorm1d(self.output_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(output_dim, output_dim, 1),
        )

        # self.bn1 = nn.BatchNorm1d(self.mid_dim)
        # self.bn2 = nn.BatchNorm1d(output_dim)
        # self.bn3 = nn.BatchNorm1d(output_dim)
        # self.bn4 = nn.BatchNorm1d(output_dim)

        # self.maxpool1 = nn.MaxPool1d(2)
        # self.relu = nn.ReLU()

    # @staticmethod
    # def knn(x1,x2,k):
    #     # x1 = x1.transpose(1,2)
    #     # x2 = x2.transpose(1,2)
    #     B, N, _ = x1.shape
    #     _, M, _ = x2.shape
    #     dist = -2 * torch.matmul(x1,x2.permute(0, 2, 1))
    #     dist += torch.sum(x1 ** 2, -1).view(B, N, 1)
    #     dist += torch.sum(x2 ** 2, -1).view(B, 1, M)
    #     knn1_indices = dist.topk(k,largest=False, sorted=False)[1]
    #     knn2_indices = dist.topk(k,dim=1,largest=False, sorted=False)[1]
    #     return knn1_indices,knn2_indices

    # @staticmethod
    # def knn_gather(x,knn_ind):
    #     # x = x.transpose(1,2)
    #     k = knn_ind.shape[-1]
    #     B, N, E = x.shape
    #     return x.unsqueeze(1).expand(B,N,N,E).gather(2,knn_ind.unsqueeze(-1).expand(B,N,k,E))

    def forward(self,x_in):
        x = self.first_mlp(x_in.transpose(1,2)).transpose(1,2).contiguous()
        # x_in = x_in.transpose(1,2).contiguous()

        _,x_cat = sample_and_group(x_in.shape[1]//2,self.k,x_in,x)
        x = torch.max(x_cat,dim=-2)[0].transpose(1,2).contiguous()

        x = self.last_mlp(x)
        return x.transpose(1,2).contiguous()


class PCup(nn.Module):
    def __init__(self, num_dense=4096, latent_dim=256, out_dim=256, grid_size=4):
        super(PCup, self).__init__()
        self.latent_dim = latent_dim
        self.num_dense = num_dense
        self.grid_size = grid_size
        self.out_dim = out_dim
        self.num_coarse = self.num_dense // (self.grid_size ** 2)

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.num_coarse)
        )

        self.final_conv = nn.Sequential(
            nn.Conv1d(self.latent_dim + 3 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512,3,1)
        )

        self.final_linear = nn.Sequential(
            nn.Linear(3,64),
            nn.ReLU(inplace=True),
            nn.Linear(64,256)
        )

        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2).cuda()  # (1, 2, S)


    def forward(self,x):

        feature_global = torch.max(x,1)[0]
        B, _ = feature_global.shape
        
        coarse = self.mlp(feature_global).reshape(-1, self.num_coarse, 3)                    # (B, num_coarse, 3), coarse point cloud
        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)             # (B, num_coarse, S, 3)
        point_feat = point_feat.reshape(-1, self.num_dense, 3).transpose(2, 1)               # (B, 3, num_fine)
        
        seed = self.folding_seed.unsqueeze(2).expand(B, -1, self.num_coarse, -1)             # (B, 2, num_coarse, S)
        seed = seed.reshape(B, -1, self.num_dense)                                           # (B, 2, num_fine)
        
        feature_global = feature_global.unsqueeze(2).expand(-1, -1, self.num_dense)          # (B, 1024, num_fine)
        feat = torch.cat([feature_global, seed, point_feat], dim=1)                          # (B, 1024+2+3, num_fine)
        fine = self.final_conv(feat) + point_feat                                            # (B, 3, num_fine), fine point cloud
        final = self.final_linear(fine.transpose(1,2))
        return final


class UDUModel(nn.Module):
    def __init__(self, input_dim=3, num_points=1024, k=10):
        super(UDUModel, self).__init__()
        self.input_dim = input_dim
        self.num_points = num_points

        self.enc1_block1 = PCdown(self.input_dim,64,k)
        self.enc1_block2 = PCdown(64,256,k)

        self.dec_down1 = PCup(1024,256,256)
        self.dec_down2 = PCup(4096,256,256)

        self.enc2_block1 = PCdown(256,64,k,type='enc2')
        self.enc2_block2 = PCdown(64,self.input_dim,k,type='enc2',last=True)



    def forward(self,x):
        x = self.enc1_block1(x)
        x = self.enc1_block2(x)

        x = self.dec_down1(x)
        x = self.dec_down2(x)
        
        x = self.enc2_block1(x)
        x = self.enc2_block2(x)
        return x


if __name__ == '__main__':
    model = UDUModel().cuda()
    x = torch.randn(32,1024,3).cuda()
    # print(model(x).shape)
    summary(model,(1024,3))