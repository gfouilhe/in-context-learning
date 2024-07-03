from collections import OrderedDict
import re
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm.notebook import tqdm
from tasks import PolynomialRegression, ChebyshevPolynomialRegression
import numpy as np
from eval import get_run_metrics, read_run_dir, get_model_from_run
from plot_utils import basic_plot, collect_results, relevant_model_names
import datetime
# run_dir = "/users/p22034/fouilhe/in-context-learning/src/models"
run_dir = "/users/p22034/fouilhe/in-context-learning/models/"

# task = "affine_regression"

task = "polynomial_regression"

run_id = "1d_3layers_deg_1_61points"

run_path = os.path.join(run_dir, task, run_id)
recompute_metrics = False

if recompute_metrics:
    get_run_metrics(run_path)  # these are normally precomputed at the end of training

def valid_row(r):
    return r.task == task and r.run_id == run_id


from samplers import get_data_sampler
from tasks import get_task_sampler

print("run_path : ", run_path)
model, conf = get_model_from_run(run_path,with_cpu=True)

# print("Model : ", model)
n_dims = conf.model.n_dims
print("n_dims", n_dims)
# n_dims = 2
# print("ndims", n_dims)
batch_size = conf.training.batch_size
# batch_size = 1
# print("batch_size", batch_size)
data_sampler = get_data_sampler(conf.training.data, n_dims)
task_sampler = get_task_sampler(
    # "quadratic_regression",
    # "linear_regression",
    # "toy_linear_regression",
    # "toy_quadratic_regression",
    "polynomial_regression",
    # "toy_quadratic_regression",
    # "toy_affine_regression",
    # "toy_polynomial_regression",
    # "chebyshev_polynomial_regression",
    n_dims,
    batch_size,
    **conf.training.task_kwargs
)

task = task_sampler(max_dim=1)

if isinstance(task, PolynomialRegression):
    coefficients = task.coefficients
    polynomial_function = None
elif isinstance(task, ChebyshevPolynomialRegression):
    polynomial_function = task.polynomial_func
    coefficients = None
else:
    coefficients = None
    polynomial_function = None




xs = data_sampler.sample_xs(b_size=batch_size, n_points=conf.training.curriculum.points.end-1)
ys = task.evaluate(xs)


print("shapes xs, ys : ",xs.shape, ys.shape)

plt.figure()
# plt.scatter(xs[0,:,0], ys[0], label="data",s=0.1)

x = xs[:,1:,0]
# flatten x along the batch dimension
x = x.flatten()
x = x.sort().values


if coefficients is not None:
    batch_size, n_dims, max_dim_p1 = coefficients.shape
    # shape = (batch_size,n_dims, max_dim + 1)
    y = torch.zeros(x.shape)
    for i in range(max_dim_p1):
        a = coefficients[0,0,i]
        print("a : ", a)
        y += a * x**(max_dim_p1 - i - 1)
    # plt.plot(x, y, label="ground truth",color='green',linestyle='--',linewidth=0.3)
    plt.scatter(x, y, label="ground truth",color='green',s=0.3)
    plt.xlabel("x")
    plt.ylabel("y")

if polynomial_function is not None:
    y = torch.zeros(x.shape)
    for i in range(len(y)):
        y[i] = polynomial_function(x[i])
    plt.scatter(x, y, label="ground truth",color='green',s=0.3)
    plt.xlabel("x")
    plt.ylabel("y")


# lower_bound = 1*torch.ones(n_dims)
# upper_bound = 2*torch.ones(n_dims)
# data_sampler = get_data_sampler("uniform", n_dims, lower=lower_bound, upper=upper_bound)
data_sampler = get_data_sampler(conf.training.data, n_dims)
x_test = data_sampler.sample_xs(b_size=batch_size, n_points=40)
# x_test = torch.sort(x_test, dim=0).values
y_test = task.evaluate(x_test)

print("x_test : ", x_test[:,:,:].shape)
print("y_test : ", y_test.shape)
# plt.scatter(x_test[0,:,0], y_test[0], label="test data",color='red',s=0.3)

# a12 = (y_test[:,0] - y_test[:,1])/ (x_test[:,0,0] - x_test[:,1,0])
# a23 = (y_test[:,1] - y_test[:,2])/ (x_test[:,1,0] - x_test[:,2,0])
# a13 = (y_test[:,0] - y_test[:,2])/ (x_test[:,0,0] - x_test[:,2,0])

# a12 = (y_test[0,:] - y_test[1,:])/ (x_test[0,:,0] - x_test[1,:,0])
# a23 = (y_test[1,:] - y_test[2,:])/ (x_test[1,:,0] - x_test[2,:,0])
# a13 = (y_test[0,:] - y_test[2,:])/ (x_test[0,:,0] - x_test[2,:,0])

# print("a12 : ", a12)
# print("a23 : ", a23)
# print("a13 : ", a13)
# print("y1 - a12 x1:", y_test[0,:] - a12*x_test[0,:,0])
# print("y2 - a12 x1:", y_test[1,:] - a12*x_test[0,:,0])
predictions = np.zeros(y_test.shape)
for i in range(x_test.shape[1]): # 0 to select the first example in each batch
    x = x_test[:,i,:].unsqueeze(1)
    y = y_test[:,i].unsqueeze(1)

    # plt.scatter(x, y, label="test data",color='red',s=0.3)
    x_s_and_test = torch.cat((xs, x), dim=1)
    y_s_and_test = torch.cat((ys, y), dim=1)

    assert x_s_and_test.shape[1] == xs.shape[1] + 1, "The number of points should be equal to the number of points in the training set"

    with torch.no_grad():
        pred = model(x_s_and_test, y_s_and_test)

    pred_on_iid_data = pred[:,:xs.shape[1]]
    pred_on_new_data = pred[:,xs.shape[1]:]
    predictions[:,i] = pred_on_new_data[:,0]

print("predictions : ", predictions.shape)
plt.scatter(x_test[0,:,0], predictions[0], label="prediction on test",color='red',marker="x",s=3)

print("tested on a polynom of max dim :", task.max_dim)
# plt.ylim(-1,1)
# plt.xlim(-1,1)
# # plt.xlim(-5, 5)
plt.legend()

plt.show()
plt.savefig(f"/users/p22034/fouilhe/in-context-learning/figures/{run_id}_on_{task.max_dim}.png")
plt.close()


# baseline = {
#     "linear_regression": 1,
#     "sparse_linear_regression": sparsity,
#     "relu_2nn_regression": n_dims,
#     "decision_tree": 1,
# }[conf.training.task]

# # plt.plot(loss.mean(axis=0), lw=2, label="Transformer")
# # plt.axhline(baseline, ls="--", color="gray", label="zero estimator")
# plt.xlabel("# in-context examples")
# plt.ylabel("squared error")
# plt.legend()
# plt.show()
# plt.savefig("/users/p22034/fouilhe/in-context-learning/squared_error.png")