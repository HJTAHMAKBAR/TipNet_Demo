class params:
    def __init__(self):
        # General Parameters
        self.cat = 'all'

        # DGCNN parameters
        self.k = 20  # EdgeConv的k个近邻
        self.emb_dims = 256     # 点云编码后特征维度256

        #
