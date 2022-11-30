import numpy as np
from sympy import Symbol

from slimfit import Model

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
gt = {
    "mu_A": 0.23,
    "mu_B": 0.55,
    "mu_C": 0.92,
    "sigma_A": 0.1,
    "sigma_B": 0.1,
    "sigma_C": 0.1,
    "sigma_D": 0.2,
    "c_A": 0.22,
    "c_B": 0.53,
    "c_C": 0.25,
}

np.random.seed(43)
N = 1000
states = ["A", "B", "C"]
xdata = np.concatenate(
    [
        np.random.normal(loc=gt[f"mu_{s}"], scale=gt[f"sigma_{s}"], size=int(N * gt[f"c_{s}"]))
        for s in states
    ]
)

np.random.shuffle(xdata)
data = {"x": xdata.reshape(-1, 1)}

# %%
guess = {
    "mu_A": 0.2,
    "mu_B": 0.4,
    "mu_C": 0.7,
    "sigma_A": 0.1,
    "sigma_B": 0.1,
    "sigma_C": 0.1,
    # "c_A": 0.33,
    # "c_B": 0.33,
    # "c_C": 0.33,
}

#%%

y = data["x"] * np.random.rand(3).reshape(1, 3)
y.shape

# %%
clear_symbols()

shape = (1, 3)
mu = symbol_matrix(name="mu", shape=shape, suffix=states)
sigma = symbol_matrix(name="sigma", shape=shape, suffix=states)
c = symbol_matrix(name="c", shape=shape, suffix=states)
gmm = GMM(Symbol("x"), mu, sigma)
model = Model({Symbol("p"): gmm})

#%%
# model.symbols
#%%
parameters = Parameters.from_symbols(gmm.symbols, guess)

# to_numerical(expr, parameters)
#%%

num_model = model.to_numerical(parameters, data)
num_model(**guess)["p"].shape

# components: list[tuple[Symbol, NumExprBase]] = []  # todo tuple LHS as variable
#
# for lhs, rhs in num_model.items():
#     if isinstance(rhs, Mul):
#         components += [(lhs, elem) for elem in rhs.elements]
#     else:
#         components.append((lhs, rhs))
# components
#
# #%%
#
# def overlapping_model_parameters(
#     model_callables: list[tuple[FitSymbol, NumExprBase]]
# ) -> list[NumericalModel]:
#
#     seen_models = []
#     seen_sets = []
#     for lhs, mc in model_callables:
#         items = set(mc.free_parameters.keys())
#
#         found = False
#         # look for sets of parameters we've seen so far, if found, append to the list of sets
#         for i, s in enumerate(seen_sets):
#             if items & s:
#                 s |= items  # add in additional items
#                 seen_models[i].append((lhs, mc))
#                 found = True
#         if not found:
#             seen_sets.append(items)
#             seen_models.append([(lhs, mc)])
#
#     # Next, piece together the dependent model parts as Model objects, restoring original multiplications
#     sub_models = []
#     for components in seen_models:
#         model_dict = defaultdict(list)
#         for lhs, rhs in components:
#             model_dict[lhs].append(rhs)
#
#         model_dict = {
#             lhs: rhs[0] if len(rhs) == 1 else Mul(*rhs) for lhs, rhs in model_dict.items()
#         }
#         sub_models.append(NumericalModel(model_dict))
#
#     return sub_models
#
#
# #%%
#
# fit = Fit(model, parameters, data, loss=LogSumLoss(sum_axis=1))
# result = fit.execute(minimizer=LikelihoodOptimizer, )

# # Compare fit result with ground truth parameters
# for k, v in result.parameters.items():
#     print(f"{k:5}: {v:10.2}, ({gt[k]:10.2})")
#
# # %%
# # repeat the fit with one of the parameters fixed
# # Parameter("mu_A", value=0.2, fixed=True)
# # Parameter("sigma_B", value=0.13, fixed=True)
# # #%%
# # result_fixed = fit.execute(guess=guess, minimizer=LikelihoodOptimizer, loss=LogSumLoss(sum_axis=1))
# #
# # for k, v in result_fixed.parameters.items():
# #     print(f"{k:5}: {v:10.2}, ({gt[k]:10.2})")
