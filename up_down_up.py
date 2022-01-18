import torch
from torch import nn
from torch.nn import MultiheadAttention
import torch.nn.functional as F
from torchsummary import summary
# from pc_transforms import compute_dist_mat

class PCdown(nn.Module):
    def __init__(self, input_dim=3, output_dim=64):
        super(PCdown, self).__init__()
        self.input_dim = input_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv1 = nn.Conv1d(input_dim, output_dim//2, kernel_size=3,padding=1)
        self.conv2 = nn.Conv1d(output_dim//2, output_dim, kernel_size=1, padding=0)
        # self.conv3 = nn.Conv1d(output_dim//2, output_dim, kernel_size=3, padding=0)

        self.bn1 = nn.BatchNorm1d(output_dim//2)
        self.bn2 = nn.BatchNorm1d(output_dim)

        self.maxpool1 = nn.MaxPool1d(2)

        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x_pool = self.maxpool1(x)
        return x_pool,x

class KAttDec(nn.Module):
    def __init__(self,input_enc,num_heads,k,att=True):
        super(KAttDec, self).__init__()
        self.k = k
        self.att = att
        self.attention = nn.MultiheadAttention(input_enc*2,num_heads,batch_first=True)

    @staticmethod
    def knn(x,k):
        x = x.transpose(1,2)
        B, N, E = x.shape
        dist = -2 * torch.matmul(x,x.permute(0, 2, 1))
        dist += torch.sum(x ** 2, -1).view(B, N, 1)
        dist += torch.sum(x ** 2, -1).view(B, 1, N)
        # dist = compute_dist_mat(x,x)
        knn_indices = dist.topk(k,largest=False, sorted=False)[1]
        output = torch.mean(x.unsqueeze(1).expand(B,N,N,E).gather(2,knn_indices.unsqueeze(-1).expand(B,N,k,E)),2)
        return output.transpose(1,2)
    
    def forward(self,*argv):
        x = argv[0]
        x_knn = self.knn(x,self.k)
        x = torch.cat([x.transpose(1,2), x_knn.transpose(1,2)], dim=-1).view(-1, 2*x.shape[-1], x.shape[1])
        x = x.transpose(1,2)
        # print(x.shape,'x in kattdec before att')
        if self.att:
            x_enc = argv[1]
            x_enc,_ = self.attention(x,x_enc,x_enc)
            x = torch.cat([x,x_enc],dim=1)
        # print(x.shape,'x in kattdec after att')
        return x


class PCup(nn.Module):
    def __init__(self, input_dim=3, output_dim=64):
        super(PCup, self).__init__()
        self.input_dim = input_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        if input_dim>output_dim:
            mid_dim = output_dim
        else:
            mid_dim = input_dim

        self.conv1 = nn.Conv1d(input_dim, mid_dim, kernel_size=3,padding=1)
        self.conv2 = nn.Conv1d(mid_dim, output_dim, kernel_size=1, padding=0)
        # self.conv3 = nn.Conv1d(output_dim//2, output_dim, kernel_size=3, padding=0)

        self.bn1 = nn.BatchNorm1d(mid_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class PCfc(nn.Module):
    def __init__(self, input_dim, output_dim, layer='conv',pool=True):
        super(PCfc, self).__init__()
        self.layer = layer
        self.pool = pool

        if layer == 'conv':
            self.conv1 = nn.Conv1d(input_dim, input_dim//2, kernel_size=1,padding=0)
            self.conv2 = nn.Conv1d(input_dim//2, output_dim, kernel_size=1, padding=0)
            self.maxpool = nn.MaxPool1d(2)
            self.bn_cnn1 = nn.BatchNorm1d(input_dim//2)
            self.bn_cnn2 = nn.BatchNorm1d(output_dim)

        elif layer == 'linear':
            self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=1,padding=0)
            self.conv2 = nn.Conv1d(32, 3, kernel_size=1, padding=0)
            self.linear1 = nn.Linear(2048,1024)
            self.bn_cnn1 = nn.BatchNorm1d(32)
            self.bn5 = nn.BatchNorm1d(output_dim)
            # self.drop1 = nn.Dropout(0.4)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()


    def forward(self,x):
        if self.layer == 'conv':
            x = self.relu(self.bn_cnn1(self.conv1(x)))
            x = self.bn_cnn2(self.conv2(x))
            if self.pool:
              x = self.maxpool(self.relu(x))
            else:
              x = self.tanh(x)
            
        elif self.layer == 'linear':
            x = self.relu(self.bn_cnn1(self.conv1(x)))
            x = self.tanh(self.bn5(self.linear1(x)))
        return x



class UDUModel(nn.Module):
    def __init__(self, input_dim=3, num_points=1024, num_heads=4):
        super(UDUModel, self).__init__()
        self.input_dim = input_dim
        self.num_points = num_points

        self.attention = nn.MultiheadAttention(num_points,num_heads,batch_first=True)
        self.enc_block1 = PCdown(self.input_dim,64)
        self.enc_block2 = PCdown(64,256)

        self.dec_down1 = PCup(512,512)
        self.dec_down2 = PCup(1024,512)
        self.dec_down3 = PCup(512,512)
        self.dec_down4 = PCup(512,256)

        self.dec_up1 = KAttDec(256,num_heads,10)
        self.dec_up2 = KAttDec(512,num_heads,10)
        self.dec_up3 = KAttDec(1024,num_heads,10,att=False)
        self.dec_up4 = KAttDec(2048,num_heads,10,att=False)

        self.attention2 = nn.MultiheadAttention(num_points*2,num_heads,batch_first=True)
        self.attention3 = nn.MultiheadAttention(num_points,num_heads,batch_first=True)
        self.enc_fc1 = PCfc(256,128,'conv')
        self.enc_fc2 = PCfc(128,32,'conv')
        self.enc_fc3 = PCfc(32,3,'conv',False)

    def forward(self,x):
        x,_ = self.attention(x,x,x)
        x1,x1_unpool = self.enc_block1(x)
        x2,x2_unpool = self.enc_block2(x1)

        x = self.dec_down1(self.dec_up1(x2,x2_unpool))
        x2 = self.dec_down2(self.dec_up2(x,x1_unpool))
        x3 = self.dec_down3(self.dec_up3(x2))
        x = self.dec_down4(self.dec_up4(x3))

        x = self.enc_fc1(x)
        x,_ = self.attention2(x,x3,x3)
        x = self.enc_fc2(x)
        x,_ = self.attention3(x,x2,x2)
        x = self.enc_fc3(x)
        return x


if __name__ == '__main__':
    model = UDUModel()
    x = torch.randn(32,3,1024)
    model(x)
    # summary(model,(3,1024))