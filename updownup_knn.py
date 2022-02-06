from cmath import tanh
import bleach
import torch
from torch import nn
from torch.nn import MultiheadAttention
import torch.nn.functional as F
from torchsummary import summary
from pc_transforms import sample_and_group

class PCdown(nn.Module):
    def __init__(self, input_dim=3, output_dim=64, k=5, type='enc1', reduction=True):
        super(PCdown, self).__init__()
        self.input_dim = input_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k = k
        self.mid_dim = input_dim//2 if type == 'enc2' else output_dim//2
        self.reduction = reduction

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
        
    def forward(self,x_in):
        if self.reduction:
            x = self.first_mlp(x_in.transpose(1,2)).transpose(1,2).contiguous()
            _,x_cat = sample_and_group(x_in.shape[1]//2,self.k,x_in,x)
            x = torch.max(x_cat,dim=-2)[0].transpose(1,2).contiguous()
        else:
            x = self.first_mlp(x_in.transpose(1,2))
            x = torch.cat([x,x_in.transpose(1,2)],dim=1)
        
        x = self.last_mlp(x)
        return x.transpose(1,2).contiguous()


class PCup(nn.Module):
    def __init__(self, num_coarse=1024, latent_dim=256,grid_size=2):
        super(PCup, self).__init__()
        self.latent_dim = latent_dim
        self.num_coarse = num_coarse
        self.grid_size = grid_size
        # self.num_coarse = self.num_dense // (self.grid_size ** 2)
        self.num_dense = self.num_coarse * (self.grid_size ** 2)

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

        # self.final_linear = nn.Sequential(
        #     nn.Linear(3,64),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(64,256)
        # )

        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2).cuda()  # (1, 2, S)


    def forward(self,feature_global,*args):

        # feature_global = torch.max(x,1)[0]
        B, _ = feature_global.shape
        
        if args:
            coarse = args[0]
        else:
            coarse = self.mlp(feature_global).reshape(-1, self.num_coarse, 3)                    # (B, num_coarse, 3), coarse point cloud
        
        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)             # (B, num_coarse, S, 3)
        point_feat = point_feat.reshape(-1, self.num_dense, 3).transpose(2, 1)               # (B, 3, num_fine)
        
        seed = self.folding_seed.unsqueeze(2).expand(B, -1, self.num_coarse, -1)             # (B, 2, num_coarse, S)
        seed = seed.reshape(B, -1, self.num_dense)                                           # (B, 2, num_fine)
        
        feature_global = feature_global.unsqueeze(2).expand(-1, -1, self.num_dense)          # (B, 1024, num_fine)
        feat = torch.cat([feature_global, seed, point_feat], dim=1)                          # (B, 1024+2+3, num_fine)
        final = self.final_conv(feat) + point_feat                                            # (B, 3, num_fine), fine point cloud
        # final = self.final_linear(final.transpose(1,2))
        return final.transpose(2,1).contiguous()


class UDUModel(nn.Module):
    def __init__(self, input_dim=3, num_points=1024, k=32):
        super(UDUModel, self).__init__()
        self.input_dim = input_dim
        self.num_points = num_points

        self.enc1_block1 = PCdown(self.input_dim,64,k,reduction=False)
        self.enc1_block2 = PCdown(64,128,k)
        self.enc1_block3 = PCdown(128,256,k)
        
        self.dec_down1 = PCup(256,256)
        self.dec_down2 = PCup(1024,256)

        self.enc2_block1 = PCdown(3,256,k)
        self.enc2_block2 = PCdown(256,128,k,type='enc2')
        self.enc2_block3 = PCdown(128,self.input_dim,k,type='enc2',reduction=False)

        self.feat_lin1 = nn.Linear(64+128+256,256)
        self.feat_lin2 = nn.Linear(64+128+256,256)

        self.attention = nn.MultiheadAttention(128,4,batch_first=True,kdim=3,vdim=3)

    def forward(self,x):
        x1 = self.enc1_block1(x)
        x2 = self.enc1_block2(x1)
        x3 = self.enc1_block3(x2)

        feat_vec1 = torch.max(x1,1)[0]
        feat_vec2 = torch.max(x2,1)[0]
        feat_vec3 = torch.max(x3,1)[0]
        feat_vec = torch.cat([feat_vec1,feat_vec2,feat_vec3],dim=-1)
        feat_vec1 = self.feat_lin1(feat_vec)
        feat_vec2 = self.feat_lin2(feat_vec)

        x1 = self.dec_down1(feat_vec1)
        x = self.dec_down2(feat_vec2,x1)
        
        x = self.enc2_block1(x)
        x = self.enc2_block2(x)
        x = x + self.attention(x,x1,x1)[0]
        x = self.enc2_block3(x)

        return x


if __name__ == '__main__':
    model = UDUModel().cuda()
    # x = torch.randn(32,1024,3).cuda()
    # print(model(x).shape)
    summary(model,(1024,3))