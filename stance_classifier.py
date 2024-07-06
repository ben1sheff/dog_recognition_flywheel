import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class StanceClassifier(nn.Module):
    def __init__(self, model_dim: (int, int)):
        super().__init__()
        conv1_feat = 10
        conv2_feat = 20
        k_size = 3
        self.pool_size = 3
        self.conv1 = nn.Conv2d(1, conv1_feat, k_size, 1)
        self.conv2 = nn.Conv2d(conv1_feat, conv2_feat, k_size, 1)
        hid_lay_size = 20 # number of nodes in hidden layer
        penult_lay_size = 20  # number of nodes in last hidden layer
        dim_exp1 = (((model_dim[0] - self.pool_size)//self.pool_size + 1)
                    , ((model_dim[1] - self.pool_size)//self.pool_size + 1))
        dim_exp2 = (((dim_exp1[0] - self.pool_size)//self.pool_size + 1)
                    , ((dim_exp1[1] - self.pool_size)//self.pool_size + 1))
        self.first_lin_layer = nn.Linear(conv2_feat * dim_exp2[0] * dim_exp2[1], hid_lay_size)
        # self.second_lin_layer = nn.Linear(hid_lay_size, hid_lay_size)
        self.third_lin_layer = nn.Linear(hid_lay_size, penult_lay_size)
        self.classifier = nn.Linear(penult_lay_size, 1)
    def forward(self, data):
        conv1_layer = F.max_pool2d(F.relu(self.conv1(data[:,None,:,:])), self.pool_size)
        conv2_layer = F.max_pool2d(F.relu(self.conv2(conv1_layer)), self.pool_size)
        first_layer = F.relu(self.first_lin_layer(torch.flatten(conv2_layer, 1)))
        # second_layer = F.relu(self.second_lin_layer(first_layer))
        # Probably should add a dropout layer, maybe one between 1 and 2 as well
        third_layer = F.relu(self.third_lin_layer(first_layer))
        return torch.squeeze(F.sigmoid(self.classifier(third_layer)))
class TestConvolution(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        num_channels = 1
        self.get_kernels()
        self.kernels = list(map(torch.Tensor, self.kernels))
        self.kernels = [kernel.view(1, 1, 3, 3).repeat(1, num_channels, 1, 1) for kernel in self.kernels]
        print(self.kernels[0].shape)
        
    def forward(self, data, i):
        data_T = torch.Tensor(data)
        return nn.functional.conv2d(data_T, self.kernels[i])
    def get_kernels(self):
        self.kernels = [
            [[ 1.,  1.,  1.],
             [-1., -1., -1.],
             [ 0.,  0.,  0.]],

            [[ 1., -1., 0],
             [ 1., -1., 0],
             [ 1., -1., 0]],
            
            [[ 1.2, 1.2,-1.],
             [ 1.2, 1.2,-1.],
             [ -1., -1.,-1.]],  
            ]
