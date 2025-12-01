import torch
x = torch.randn(3, 4)
print(x.shape)
print(x.view(4, 3).shape)
