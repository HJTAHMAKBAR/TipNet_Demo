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
    # 利用矩阵运算公式求knn的距离矩阵，计算x的点与点的特征空间距离
    # 博客参考：https://blog.csdn.net/qq_40816078/article/details/112652548
    # x (batch, feature, num)
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # 内积，(batch, num, num)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # 逐元素平方，在特征维度求和，(batch, 1, num)，后面计算会广播展开
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # 根据公式计算，(batch, num, num)，dis[i, j]表示第i点到第j点距离
    # 取距离最小的k个点的下标
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    # 博客参考：https://zhuanlan.zhihu.com/p/619921345
    # x.shape = [B, F, N]
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 将batch合并，需要将后面的序号接力递增，以保证编号唯一
    # 为了将输入x的从原先的形状[batch_size, num_points, feature_dims]展开为了[batch_size * num_points, feature_dims]
    # 所以需要对每个点云的索引加上idx_base(0, num_points, 2 * num_points, ...)
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    # 把batch里的点云pair张量拉长
    # 每k个元素为一组，分别表示第0, 1, 2, ...个点云的k个邻接点的编号
    idx = idx.view(-1)

    _, feature_dims, _ = x.size()

    # transpose和view之间需要用contiguous()，view操作需要保证内存连续
    # (batch_size, feature_dims, num_points) -> (batch_size, num_points, feature_dims)
    x = x.transpose(2, 1).contiguous()
    # (batch_size, num_points, feature_dims)  -> (batch_size * num_points, feature_dims)
    feature = x.view(batch_size * num_points, -1)
    # 每一行表示点云的特征，下标是点的编号
    feature = x.view(batch_size * num_points, -1)[idx, :]
    # 按k个元素分一类
    feature = feature.view(batch_size, num_points, k, feature_dims)
    # 在第三个维度(特征维度)上复制k份
    x = x.view(batch_size, num_points, 1, feature_dims).repeat(1, 1, k, 1)

    # 邻接点xj - xi 和 xi 拼接
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class EdgePoolingLayer(nn.Module):
    """ Dynamic Edge Pooling Layer """

    def __init__(self, in_channels, k, ratio=0.5, scoring_func="tanh", num_points=-1):
        super().__init__()
        self.in_channels = in_channels
        self.k = k
        self.ratio = ratio
        self.score_layer = nn.Conv2d(in_channels * 2, 1, kernel_size=1, bias=True)
        self.scoring_func = scoring_func
        self.num_points = num_points


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
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        # 返回k个中的最大值
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        # 池化


if __name__ == '__main__':
    x = torch.randn(2, 3, 60).cuda()
    dgcnn = DGCNN().cuda()
    dgcnn(x)
