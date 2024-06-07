from collections import OrderedDict
import re
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm.notebook import tqdm
import numpy as np
from eval import get_run_metrics, read_run_dir, get_model_from_run
from plot_utils import basic_plot, collect_results, relevant_model_names


# sns.set_theme('notebook', 'darkgrid')
# palette = sns.color_palette('colorblind')

# run_dir = "/users/p22034/fouilhe/in-context-learning/src/models"
run_dir = "/users/p22034/fouilhe/in-context-learning/models/"

# df = read_run_dir(run_dir)
# print(df)  # list all the runs in our run_dir

# task = "affine_regression"
#task = "sparse_linear_regression"
#task = "decision_tree"
#task = "relu_2nn_regression"
task = "polynomial_regression"

# run_id = "pretrained"  # if you train more models, replace with the run_id from the table above
run_id = "1d"

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
batch_size = 1
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
    n_dims,
    batch_size,
    **conf.training.task_kwargs
)

task = task_sampler(max_dim=3)
xs = data_sampler.sample_xs(b_size=batch_size, n_points=50)
ys = task.evaluate(xs)


print("shapes xs, ys : ",xs.shape, ys.shape)

plt.figure()
# plt.scatter(xs[0,:,0], ys[0], label="data",s=0.1)

x = xs[0,:,0].sort().values.unsqueeze(1).unsqueeze(2)
# x = x.view(1,64,1).repeat(batch_size,1,1)
# print(x.shape)
y = task.evaluate(x)
# print(y.shape)
# y = y[0]
# x = x[0,:,0]
# print("x : ", x)
# print("y : ", y)

# y=2.4*x
# y=2.4*x-2
# y = task.evaluate(x)
# y=2*x**2+3*x-1
# plt.plot(x, y, label="y=2.4x",color='blue')
# plt.plot(x, y, label="y=2.4x-2",color='blue')
plt.plot(x, y, label="y",color='grey',linestyle='--',linewidth=0.3)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
# plt.savefig("/users/p22034/fouilhe/in-context-learning/data.png")
# raise ValueError

# metric = task.get_metric()
# loss = metric(pred_on_iid_data, ys).numpy()

# lower_bound = 1*torch.ones(n_dims)
# upper_bound = 2*torch.ones(n_dims)
# data_sampler = get_data_sampler("uniform", n_dims, lower=lower_bound, upper=upper_bound)
data_sampler = get_data_sampler(conf.training.data, n_dims)
x_test = data_sampler.sample_xs(b_size=batch_size, n_points=80)
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

x_s_and_test = torch.cat((xs, x_test), dim=1)
y_s_and_test = torch.cat((ys, y_test), dim=1)

print("x_s_and_test : ", x_s_and_test.shape)
print("y_s_and_test : ", y_s_and_test.shape)


with torch.no_grad():
    pred = model(x_s_and_test, y_s_and_test)

pred_on_iid_data = pred[:,:xs.shape[1]]
pred_on_new_data = pred[:,xs.shape[1]:]

# with torch.no_grad():
#     pred_on_iid_data = model(xs, ys)
# # sparsity = conf.training.task_kwargs.sparsity if "sparsity" in conf.training.task_kwargs else None

# with torch.no_grad():
#     pred_on_new_data = model(x_test, y_test)

# plt.scatter(xs[0,1:,0], pred_on_iid_data[0,1:], label="prediction on train after the first batch",color='orange',marker='x',s=2)

# print("pred_on_new_data : ", pred_on_new_data)
# print("x_test : ", x_test)
# print("y_test : ", y_test)

plt.scatter(x_test[0,1:,0], pred_on_new_data[0,1:], label="prediction on test",color='red',marker="x",s=3)

# plt.ylim(-10, 10)
# plt.xlim(-5, 5)
plt.legend()
plt.show()
plt.savefig("/users/p22034/fouilhe/in-context-learning/data.png")
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