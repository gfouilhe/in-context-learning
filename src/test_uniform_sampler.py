import math

import torch

from samplers import UniformSampler, GaussianSampler

lower_bound = torch.tensor([-10.0, 0.0])
upper_bound = torch.tensor([100.0, 200.0])

sampler = UniformSampler(2,lower_bound, upper_bound)
print("UniformSampler")
print(sampler.sample_xs(10,3))

###
print("GaussianSampler")
sampler = GaussianSampler(2)

print(sampler.sample_xs(10,3))
