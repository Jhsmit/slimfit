# %% [markdown]
#
# Given known gaussian mixture model parameters, find best fit parameters for their amplitudes.
#

# %%

import numpy as np
from sympy import Symbol

from slimfit import Model

from slimfit.numerical import GMM
from slimfit.fit import Fit
from slimfit.loss import LogSumLoss
from slimfit.minimizers import LikelihoodOptimizer
from slimfit.operations import Mul
from slimfit.parameter import Parameters
from slimfit.symbols import (
    symbol_matrix,
    clear_symbols,
    get_symbols,
)

# %%
gt = {
    "mu_A": 0.23,
    "mu_B": 0.55,
    "mu_C": 0.92,
    "sigma_A": 0.1,
    "sigma_B": 0.1,
    "sigma_C": 0.1,
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
    "c_A": 0.33,
    "c_B": 0.33,
    "c_C": 0.33,
}

# %%
clear_symbols()


# %%
g_shape = (1, 3)
c_shape = (3, 1)
mu = symbol_matrix(name="mu", shape=g_shape, suffix=states)
sigma = symbol_matrix(name="sigma", shape=g_shape, suffix=states)
c = symbol_matrix(name="c", shape=c_shape, suffix=states)
gmm = GMM(Symbol("x"), mu, sigma)

# pre-evaluate the GMM part
num_gmm = gmm.numerical(**data, **gt)

# %%
# The model now only has amplitude parameters
model = Model({Symbol("p"): Mul(c, num_gmm)})

# %%
symbols = get_symbols(mu, sigma, c)
parameters = Parameters.from_symbols(symbols.values(), guess)

fit = Fit(model, parameters, data, loss=LogSumLoss(sum_axis=1))
result = fit.execute(minimizer=LikelihoodOptimizer)

# Compare fit result with ground truth parameters
for k, v in result.parameters.items():
    print(f"{k:5}: {v:10.2}, ({gt[k]:10.2})")
