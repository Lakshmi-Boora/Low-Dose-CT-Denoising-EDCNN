import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class EdgeEnhancementModule(nn.Module):
    def __init__(self, groups=8):
        super().__init__()
        self.groups = groups
        self.alpha = nn.Parameter(torch.ones(groups)) # Learnable Sobel Factor
        
        # Static Sobel Templates
        v = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        h = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        d1 = torch.tensor([[0., 1., 2.], [-1., 0., 1.], [-2., -1., 0.]]).view(1, 1, 3, 3)
        d2 = torch.tensor([[-2., -1., 0.], [-1., 0., 1.], [0., 1., 2.]]).view(1, 1, 3, 3)
        
        self.register_buffer('v_kernel', v)
        self.register_buffer('h_kernel', h)
        self.register_buffer('d1_kernel', d1)
        self.register_buffer('d2_kernel', d2)

    def forward(self, x):
        outputs = [x]
        for i in range(self.groups):
            a = self.alpha[i]
            outputs.append(F.conv2d(x, self.v_kernel * a, padding=1))
            outputs.append(F.conv2d(x, self.h_kernel * a, padding=1))
            outputs.append(F.conv2d(x, self.d1_kernel * a, padding=1))
            outputs.append(F.conv2d(x, self.d2_kernel * a, padding=1))
        return torch.cat(outputs, dim=1) # 33 Channels

class DenseConvBlock(nn.Module):
    def __init__(self, in_ch, edge_ch, out_ch=32, is_last=False):
        super().__init__()
        self.is_last = is_last
        self.fusion = nn.Conv2d(in_ch + edge_ch, out_ch, kernel_size=1)
        self.learning = nn.Conv2d(out_ch, 1 if is_last else out_ch, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, edges):
        feat = torch.cat([x, edges], dim=1)
        feat = self.relu(self.fusion(feat))
        out = self.learning(feat)
        return out if self.is_last else self.relu(out)

class EDCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.eem = EdgeEnhancementModule()
        self.blocks = nn.ModuleList()
        # First block
        self.blocks.append(DenseConvBlock(33, 0, out_ch=32))
        # Intermediate blocks
        for _ in range(6):
            self.blocks.append(DenseConvBlock(32, 33, out_ch=32))
        # Reconstruction block
        self.blocks.append(DenseConvBlock(32, 33, out_ch=32, is_last=True))

    def forward(self, x):
        edges = self.eem(x)
        feat = self.blocks[0](edges, torch.empty(0, device=x.device))
        for i in range(1, 8):
            feat = self.blocks[i](feat, edges)
        return x + feat # Global Residual Learning