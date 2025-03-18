import torch.nn as nn
import torch
import torch.nn.functional as F

class DConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        # self.args = args
        # self.k = args.k
        self.leaky_relu = 1
        self.dim = dim
        self.bn1 = nn.BatchNorm1d(self.dim)
        self.bn2 = nn.BatchNorm1d(self.dim)
        self.bn3 = nn.BatchNorm1d(self.dim)
        self.bn5 = nn.BatchNorm1d(self.dim)

        if self.leaky_relu:
            act_mod = nn.LeakyReLU
            act_mod_args = {'negative_slope': 0.2}
        else:
            act_mod = nn.ReLU
            act_mod_args = {}

        self.conv1 = nn.Sequential(nn.Conv1d(self.dim, self.dim, kernel_size=1, bias=False),
                                   self.bn1,
                                   act_mod(**act_mod_args))
        self.conv2 = nn.Sequential(nn.Conv1d(self.dim, self.dim, kernel_size=1, bias=False),
                                   self.bn2,
                                   act_mod(**act_mod_args))
        self.conv3 = nn.Sequential(nn.Conv1d(self.dim, self.dim, kernel_size=1, bias=False),
                                   self.bn3,
                                   act_mod(**act_mod_args))
        self.conv5 = nn.Sequential(nn.Conv1d(self.dim * 3, self.dim, kernel_size=1, bias=False),
                                   self.bn5,
                                   act_mod(**act_mod_args))


    def forward(self, x):
        batch_size = x.size(0)  # 64
        # x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]    # 64，768

        # x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]    # 64，768

        # x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1).unsqueeze(1)

        return x1