from collections import OrderedDict
import re
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm.notebook import tqdm

from eval import get_run_metrics, read_run_dir, get_model_from_run
from plot_utils import basic_plot, collect_results, relevant_model_names


# sns.set_theme('notebook', 'darkgrid')
# palette = sns.color_palette('colorblind')

run_dir = "/users/p22034/fouilhe/in-context-learning/src/models"

df = read_run_dir(run_dir)
# print(df)  # list all the runs in our run_dir

task = "linear_regression"
#task = "sparse_linear_regression"
#task = "decision_tree"
#task = "relu_2nn_regression"

run_id = "pretrained"  # if you train more models, replace with the run_id from the table above

run_path = os.path.join(run_dir, task, run_id)
recompute_metrics = False

if recompute_metrics:
    get_run_metrics(run_path)  # these are normally precomputed at the end of training

def valid_row(r):
    return r.task == task and r.run_id == run_id

# metrics = collect_results(run_dir, df, valid_row=valid_row)
# _, conf = get_model_from_run(run_path, only_conf=True)
# n_dims = conf.model.n_dims

# models = relevant_model_names[task]
# basic_plot(metrics["standard"], models=models)
# plt.show()
# plt.savefig("/users/p22034/fouilhe/in-context-learning/standard.png")
# plt.close()

# plot any OOD metrics
# for name, metric in metrics.items():
#     if name == "standard": continue
   
#     if "scale" in name:
#         scale = float(name.split("=")[-1])**2
#     else:
#         scale = 1.0

#     trivial = 1.0 if "noisy" not in name else (1+1/n_dims)
#     fig, ax = basic_plot(metric, models=models, trivial=trivial * scale)
#     ax.set_title(name)
    
#     if "ortho" in name:
#         ax.set_xlim(-1, n_dims - 1)
#     ax.set_ylim(-.1 * scale, 1.5 * scale)

#     plt.show()
#     plt.savefig(f"{name}.png")

from samplers import get_data_sampler
from tasks import get_task_sampler

print("run_path : ", run_path)
model, conf = get_model_from_run(run_path,with_cpu=True)
# print("Model : ", model)
n_dims = conf.model.n_dims
# print("ndims", n_dims)
batch_size = conf.training.batch_size
# print("batch_size", batch_size)
data_sampler = get_data_sampler(conf.training.data, n_dims)
task_sampler = get_task_sampler(
    # "quadratic_regression",
    "linear_regression",
    n_dims,
    batch_size,
    **conf.training.task_kwargs
)

task = task_sampler()
xs = data_sampler.sample_xs(b_size=batch_size, n_points=conf.training.curriculum.points.end)
ys = task.evaluate(xs)
# print("shapes : ",xs.shape, ys.shape)

# plt.figure()
# plt.scatter(xs[:,:,0], ys, label="data")
# plt.xlabel("x")
# plt.ylabel("y_true")
# plt.show()
# plt.savefig("/users/p22034/fouilhe/in-context-learning/data.png")
# plt.close()

# with torch.no_grad():
#     pred_on_iid_data = model(xs, ys)
sparsity = conf.training.task_kwargs.sparsity if "sparsity" in conf.training.task_kwargs else None

metric = task.get_metric()
loss = metric(pred_on_iid_data, ys).numpy()

lower_bound = torch.zeros(n_dims)
upper_bound = torch.ones(n_dims)

data_sampler = get_data_sampler("uniform", n_dims, lower=lower_bound, upper=upper_bound)
x_test = data_sampler.sample_xs(b_size=1, n_points=3)
y_test = task.evaluate(x_test)

print("x_test : ", x_test)
print("y_test : ", y_test)





# baseline = {
#     "linear_regression": 1,
#     "sparse_linear_regression": sparsity,
#     "relu_2nn_regression": n_dims,
#     "decision_tree": 1,
# }[conf.training.task]

plt.plot(loss.mean(axis=0), lw=2, label="Transformer")
# plt.axhline(baseline, ls="--", color="gray", label="zero estimator")
plt.xlabel("# in-context examples")
plt.ylabel("squared error")
plt.legend()
plt.show()
plt.savefig("/users/p22034/fouilhe/in-context-learning/squared_error.png")