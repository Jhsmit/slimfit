from collections import defaultdict

import numpy as np
import proplot as pplt
from sympy import Symbol

from slimfit import Model
from slimfit.models import NumericalModel
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
    )

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
mu = symbol_matrix(name="mu", shape=(3, 1), suffix=states)
sigma = symbol_matrix(name="sigma", shape=(3, 1), suffix=states)
c = symbol_matrix(name="c", shape=(3, 1), suffix=states)
model_dict[Symbol("p1")] = Mul(c, GMM(Symbol("x1"), mu, sigma))

states = ["B", "C", "D"]
mu = symbol_matrix(name="mu", shape=(3, 1), suffix=states)
sigma = symbol_matrix(name="sigma", shape=(3, 1), suffix=states)
c = symbol_matrix(name="c", shape=(3, 1), suffix=states)
model_dict[Symbol("p2")] = Mul(c, GMM(Symbol("x2"), mu, sigma))

model = Model(model_dict)

#%%

parameters = Parameters.from_symbols(model.symbols, guess)
parameters

# %%
num_model = model.to_numerical(parameters)
num_model.parameters

#%%
components: list[tuple[Symbol, NumExprBase]] = []  # todo tuple LHS as variable
for lhs, rhs in num_model.items():
    if isinstance(rhs, Mul):
        components += [(lhs, elem) for elem in rhs.elements]
    else:
        components.append((lhs, rhs))
components

#%%
lhs, num_expr = components[1]
num_expr.parameters


Matrix should be normal sympy; convert to numexpr
GMM should be composite

#%%

model_callables = components

seen_models = []
seen_sets = []
for lhs, num_expr in model_callables:
    param_set = set(num_expr.free_parameters.keys())

    found = False
    # look for sets of parameters we've seen so far, if found, append to the list of sets
    for i, s in enumerate(seen_sets):
        # some of the parameters in this numexpr
        if param_set & s:
            s |= param_set  # add in additional items
            seen_models[i].append((lhs, num_expr))
            found = True
    if not found:
        seen_sets.append(param_set)
        seen_models.append([(lhs, num_expr)])

seen_sets, seen_models

#%%
# Next, piece together the dependent model parts as Model objects, restoring original multiplications
sub_models = []
for param_set, model_components in zip(seen_sets, seen_models):
    model_dict = defaultdict(list)
    for lhs, rhs in model_components:
        model_dict[lhs].append(rhs)

    model_dict = {
        lhs: rhs[0] if len(rhs) == 1 else Mul(*rhs) for lhs, rhs in model_dict.items()
    }
    print(param_set)
    print(model_components)
    sub_models.append(NumericalModel(model_dict))




#%%

#
# fit = Fit(model_dict, **data)
# result = fit.execute(guess=guess, minimizer=LikelihoodOptimizer, loss=LogSumLoss(sum_axis=1))
# print(result.gof_qualifiers)
#
# # Compare fit result with ground truth parameters
# for k, v in result.parameters.items():
#     print(f"{k:5}: {v:10.2}, ({gt[k]:10.2})")
#
# # %%
#
# x_point = np.linspace(-0.5, 1.5, num=250)
# eval = fit.model(x1=x_point, x2=x_point, **result.parameters)
#
# fig, axes = pplt.subplots(ncols=2)
# colors = {"A": "g", "B": "b", "C": "cyan", "D": "magenta"}
#
# for i, ax in enumerate(axes):
#
#     ax.hist(data[f"x{i + 1}"], bins="fd", density=True, color="gray")
#     ax.plot(x_point, eval[f"p{i + 1}"].sum(axis=1))
#     for j, state in enumerate(all_states[i]):
#         print(j, state)
#         ax.plot(x_point, eval[f"p{i + 1}"][:, j], color=colors[state])
#     ax.format(title=f"Dataset {i + 1}")
# pplt.show()
#
#
# #%%
#
# # Fitting only dataset two
# model_ds2 = {Probability("p2"): model_dict[Probability("p2")]}
#
# fit = Fit(model_ds2, x2=data["x2"])
# result = fit.execute(guess=guess, minimizer=LikelihoodOptimizer, loss=LogSumLoss(sum_axis=1))
#
# print(result.gof_qualifiers)
#
# # Compare fit result with ground truth parameters
# for k, v in result.parameters.items():
#     print(f"{k:5}: {v:10.2}, ({gt[k]:10.2})")
