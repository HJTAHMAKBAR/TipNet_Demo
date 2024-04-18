import torch
import torch.nn as nn
from torchvision import models


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        base = models.resnet18()
        # *list将列表的元素拆开，分别逐个传入，
        # resnet基本结构Input -> Conv1 -> ResBlock1 -> ResBlock2 -> ... -> ResBlockN -> AvgPool -> FC -> Output
        # 去掉网络结构的最后三层，只得到图片的特征向量
        self.base = nn.Sequential(*list(base.children())[:-3])
        # 获取图像特征大小
        in_features = base.fc.in_features

    def forward(self, x):
        x = self.base(x)
        x = x.view(x.size(0), 256, -1)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(2, 3, 224, 224).to(device)
    model = ResNet().to(device)
    out = model(x)
    print('out_size:', out.shape)
