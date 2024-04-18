import torch.nn as nn



class TipNet(nn.Module):

    def __init__(self):
        super(TipNet, self).__init__()

        # Encoders for text, image and point cloud
        self.pc_encoder = ''
        self.im_encoder = ''