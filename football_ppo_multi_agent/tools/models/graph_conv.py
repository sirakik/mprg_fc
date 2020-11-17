import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, A, s_kernel_size):
        super(GraphConv, self).__init__()
        self.A = A
        self.s_kernel_size = s_kernel_size
        self.conv = nn.Conv1d(in_channels = in_channels,
                              out_channels=out_channels * s_kernel_size,
                              kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        n, kc, v = x.size()
        x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, v)
        x = torch.einsum('nkcv,kvw->ncw', (x, self.A))

        return x.contiguous()
