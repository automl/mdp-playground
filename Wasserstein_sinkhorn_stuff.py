# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.autograd import grad

use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

from geomloss import SamplesLoss
# from pykeops.torch.cluster import grid_cluster, cluster_ranges_centroids

num_points = 3
num_dims = 2
A_i = np.random.rand(num_points, 1)
A_i = A_i / np.sum(A_i)
X_i = np.random.rand(num_points, num_dims) - np.ones((num_points, num_dims))/2
B_j = np.random.rand(num_points, 1)
B_j = B_j / np.sum(B_j)
Y_j = np.random.rand(num_points, num_dims) - np.ones((num_points, num_dims))/2

print(A_i.shape, X_i.shape, B_j.shape, Y_j.shape)
print(A_i, X_i, B_j, Y_j)

#plot variations across num_points, num_dims, p = 1 or 2, blur, diameter,



scaling, Nits = .5, 9
cluster_scale = .1 if not use_cuda else .05
i = 3
blur = scaling**i

if blur > cluster_scale:
    print('Calculating Sinkhorn divergences over coarse clusters. blur, cluster_scale =', blur, cluster_scale)
else:
    print('Calculating Sinkhorn divergences over actual points. blur, cluster_scale =', blur, cluster_scale)

# Create a copy of the data...
A_i_torch = torch.from_numpy(A_i).type(dtype)
X_i_torch = torch.from_numpy(X_i).contiguous().type(dtype)
B_j_torch = torch.from_numpy(B_j).type(dtype)
Y_j_torch = torch.from_numpy(Y_j).contiguous().type(dtype)
a_i, x_i = A_i_torch.clone(), X_i_torch.clone()
b_j, y_j = B_j_torch.clone(), Y_j_torch.clone()

# And require grad:
a_i.requires_grad = True
x_i.requires_grad = True
b_j.requires_grad = True

# Compute the loss + gradients:
# Loss_p1 = SamplesLoss("sinkhorn", p=1, blur=blur, diameter=1., cluster_scale = cluster_scale,
#                         scaling=scaling, backend="multiscale", verbose=True)
# loss_p1 = Loss_p1(a_i, x_i, b_j, y_j)
Loss_p2 = SamplesLoss("sinkhorn", p=2, blur=blur, diameter=1., cluster_scale = cluster_scale,
                        scaling=scaling, backend="multiscale", verbose=True)
loss_p2 = Loss_p2(a_i, x_i, b_j, y_j)


# print("Loss_p1 =", Loss_p1, "Loss_p2 =", Loss_p2)
# print("loss_p1 =", loss_p1, "loss_p2 =", loss_p2)



# import matplotlib.pyplot as plt
# plt.figure(figsize=((12, 9)))
#
# size_scale = 2000
# ax = plt.scatter(X_i[:, 0], X_i[:, 1], s=size_scale * A_i, c='blue')
# ax = plt.scatter(Y_j[:, 0], Y_j[:, 1], s=size_scale * B_j, c='red')
#
#
# plt.tight_layout()
# plt.show()


i = 4
blur = scaling**i

if blur > cluster_scale:
    print('Calculating Sinkhorn divergences over coarse clusters. blur, cluster_scale =', blur, cluster_scale)
else:
    print('Calculating Sinkhorn divergences over actual points. blur, cluster_scale =', blur, cluster_scale)
Loss_p2 = SamplesLoss("sinkhorn", p=2, blur=blur, diameter=1., cluster_scale = cluster_scale,
                        scaling=scaling, backend="multiscale")
loss_p2 = Loss_p2(a_i, x_i, b_j, y_j)
