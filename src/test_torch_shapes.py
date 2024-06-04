import torch

t1 = torch.randn([64, 41, 20])
t2 = torch.randn([64, 20, 1])
t3 = torch.randn([64, 20])

print(t1.shape)
print(t2.shape)
print(t3.shape)
print((t1 @ t2).shape)
print((t1 @ t3).shape)
print((t2 @ t1).shape)
print((t3 @ t1).shape)
