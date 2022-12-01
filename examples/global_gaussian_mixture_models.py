from collections import defaultdict

import numpy as np
import proplot as pplt
from sympy import Symbol

from slimfit import Model
# from slimfit.models import NumericalModel
from slimfit.numerical import GMM, NumExprBase
from slimfit.fit import Fit
from slimfit.loss import LogSumLoss
from slimfit.minimizer import LikelihoodOptimizer
from slimfit.operations import Mul
from slimfit.parameter import Parameters
from slimfit.symbols import (
    symbol_matrix,
    clear_symbols,
)

# %%

all_states = ["ABC", "BCD"]

gt = {
    "mu_A": 0.23,
    "mu_B": 0.55,
    "mu_C": 0.92,
    "mu_D": 0.32,
    "sigma_A": 0.1,
    "sigma_B": 0.1,
    "sigma_C": 0.1,
    "sigma_D": 0.2,
    "c_A": 0.22,
    "c_B": 0.53,
    "c_C": 0.25,
    "c_D": 0.22,
}

np.random.seed(43)

vars = ["x1", "x2"]
data = {}
Ns = [1000, 1500]
for st, var, N in zip(all_states, vars, Ns):
    data[var] = np.concatenate(
        [
            np.random.normal(loc=gt[f"mu_{s}"], scale=gt[f"sigma_{s}"], size=int(N * gt[f"c_{s}"]))
            for s in st
        ]
    ).reshape(-1, 1)

#%%
data

# %%
guess = {
    "mu_A": 0.2,
    "mu_B": 0.4,
    "mu_C": 0.7,
    "mu_D": 0.15,
    "sigma_A": 0.1,
    "sigma_B": 0.1,
    "sigma_C": 0.1,
    "sigma_D": 0.1,
    "c_A": 0.33,
    "c_B": 0.33,
    "c_C": 0.33,
    "c_D": 0.33,
}

# %%
clear_symbols()
model_dict = {}

states = ["A", "B", "C"]
g_shape = (1, 3)
c_shape = (3, 1)
mu = symbol_matrix(name="mu", shape=g_shape, suffix=states)
sigma = symbol_matrix(name="sigma", shape=g_shape, suffix=states)
c = symbol_matrix(name="c", shape=c_shape, suffix=states)
model_dict[Symbol("p1")] = Mul(c, GMM(Symbol("x1"), mu, sigma))

states = ["B", "C", "D"]
mu = symbol_matrix(name="mu", shape=g_shape, suffix=states)
sigma = symbol_matrix(name="sigma", shape=g_shape, suffix=states)
c = symbol_matrix(name="c", shape=c_shape, suffix=states)
model_dict[Symbol("p2")] = Mul(c, GMM(Symbol("x2"), mu, sigma))

model = Model(model_dict)
model
#%%

# create parameters from the symbols in the model if they are in the guess dictionary
parameters = Parameters.from_symbols(model.symbols, gt)
fit = Fit(model, parameters, data, loss=LogSumLoss(sum_axis=1))
result = fit.execute(minimizer=LikelihoodOptimizer)

# Compare fit result with ground truth parameters
for k, v in result.parameters.items():
    print(f"{k:5}: {v:10.2}, ({gt[k]:10.2})")

# %%

x_point = np.linspace(-0.5, 1.5, num=250).reshape(-1, 1)
eval_data = {'x1': x_point, 'x2': x_point}
num_model = model.to_numerical(parameters, eval_data)


#%%

ans = num_model(**result.parameters)
gt_ans = num_model(**gt)

fig, axes = pplt.subplots(ncols=2)
colors = {"A": "g", "B": "b", "C": "cyan", "D": "magenta"}

for i, ax in enumerate(axes):

    ax.hist(data[f"x{i + 1}"].squeeze(), bins="fd", density=True, color="gray")
    ax.plot(x_point, ans[f"p{i + 1}"].sum(axis=1), color='k')
    for j, state in enumerate(all_states[i]):
        ax.plot(x_point, ans[f"p{i + 1}"][:, j], color=colors[state])
        ax.plot(x_point, gt_ans[f"p{i + 1}"][:, j], color=colors[state], linestyle='--')
    ax.format(title=f"Dataset {i + 1}")
pplt.show()
#
#
# #%%
#
# Fitting only dataset two
model_ds2 = Model({Symbol("p2"): model_dict[Symbol("p2")]})
#
model_ds2.symbols

#%%
guess = {
    "mu_B": 0.4,
    "mu_C": 0.7,
    "mu_D": 0.15,
    "sigma_B": 0.1,
    "sigma_C": 0.1,
    "sigma_D": 0.1,
    "c_B": 0.33,
    "c_C": 0.33,
    "c_D": 0.33,
}

parameters = Parameters.from_symbols(model_ds2.symbols, guess)

#%%

fit = Fit(model_ds2, parameters, {'x2': data['x2']}, loss=LogSumLoss(sum_axis=1))
result = fit.execute(minimizer=LikelihoodOptimizer)

print(result.gof_qualifiers)
#
# Compare fit result with ground truth parameters
for k, v in result.parameters.items():
    print(f"{k:5}: {v:10.2}, ({gt[k]:10.2})")
