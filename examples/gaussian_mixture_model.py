import numpy as np
from slimfit import Model

from slimfit.numerical import GMM
from slimfit.fit import Fit
from slimfit.loss import LogSumLoss
from slimfit.minimizer import LikelihoodOptimizer
from slimfit.operations import Mul
from slimfit.symbols import (
    parameter_matrix,
    Variable,
    Probability,
    clear_symbols,
    Parameter,
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
data = {"x": xdata}

# %%
guess = {
    "mu_A": 0.2,
    "mu_B": 0.4,
    "mu_C": 0.7,
    "sigma_A": 0.1,
    "sigma_B": 0.1,
    "sigma_C": 0.1,
    "c_A": 0.33,
    "c_B": 0.33,
    "c_C": 0.33,
    "c_D": 0.33,
}

# %%
clear_symbols()

mu = parameter_matrix(name="mu", shape=(3, 1), suffix=states, rand_init=True)
sigma = parameter_matrix(name="sigma", shape=(3, 1), suffix=states, rand_init=True)
c = parameter_matrix(name="c", shape=(3, 1), suffix=states, norm=True)
model = Model({Probability("p"): Mul(c, GMM(Variable("x"), mu, sigma))})

# %%
fit = Fit(model, **data)
result = fit.execute(guess=guess, minimizer=LikelihoodOptimizer, loss=LogSumLoss(sum_axis=1))

# Compare fit result with ground truth parameters
for k, v in result.parameters.items():
    print(f"{k:5}: {v:10.2}, ({gt[k]:10.2})")

# %%
# repeat the fit with one of the parameters fixed
Parameter("mu_A", value=0.2, fixed=True)
Parameter("sigma_B", value=0.13, fixed=True)
#%%
result_fixed = fit.execute(guess=guess, minimizer=LikelihoodOptimizer, loss=LogSumLoss(sum_axis=1))

for k, v in result_fixed.parameters.items():
    print(f"{k:5}: {v:10.2}, ({gt[k]:10.2})")
