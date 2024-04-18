import os
import sys
import inspect
import torch
import torch.nn as nn

# import config from the parent directory
# 获取当前文件的目录绝对路径
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from config import params

args = params()


def knn(x, k):
    # 利用矩阵运算求knn的距离矩阵，计算x的点与点的距离
    # x (batch, feature, num)
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # 内积
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # 逐元素平方，在特征维度求和
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]        # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    # x.shape = [B, F, N]
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DGCNN(nn.Module):

    def __init__(self, output_channels=512):
        super(DGCNN, self).__init__()

        self.k = args.k
        self.output_channels = args.emb_dims
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv5 = nn.Sequential(nn.Conv2d(256, self.output_channels, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        # x: [B, F, N]
        x = x


if __name__ == '__main__':
    x = torch.Tensor([[[1, 2],
         [ 2,  3],
         [3,  4]],

        [[4,  5],
         [5, 6],
         [ 6, 7]]])
    print(x)
    # x = torch.sum(x ** 2, dim=1, keepdim=True)
    print(torch.matmul(x.transpose(1, 2), x))
