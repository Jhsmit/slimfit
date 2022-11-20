import numpy as np
import proplot as pplt

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
mu = parameter_matrix(name="mu", shape=(3, 1), suffix=states, rand_init=True)
sigma = parameter_matrix(name="sigma", shape=(3, 1), suffix=states, rand_init=True)
c = parameter_matrix(name="c", shape=(3, 1), suffix=states, norm=True)
model_dict[Probability("p1")] = Mul(c, GMM(Variable("x1"), mu, sigma))

states = ["B", "C", "D"]
mu = parameter_matrix(name="mu", shape=(3, 1), suffix=states, rand_init=True)
sigma = parameter_matrix(name="sigma", shape=(3, 1), suffix=states, rand_init=True)
c = parameter_matrix(name="c", shape=(3, 1), suffix=states, norm=True)
model_dict[Probability("p2")] = Mul(c, GMM(Variable("x2"), mu, sigma))

# %%

fit = Fit(model_dict, **data)
result = fit.execute(guess=guess, minimizer=LikelihoodOptimizer, loss=LogSumLoss(sum_axis=1))
print(result.gof_qualifiers)

# Compare fit result with ground truth parameters
for k, v in result.parameters.items():
    print(f"{k:5}: {v:10.2}, ({gt[k]:10.2})")

# %%

x_point = np.linspace(-0.5, 1.5, num=250)
eval = fit.model(x1=x_point, x2=x_point, **result.parameters)

fig, axes = pplt.subplots(ncols=2)
colors = {"A": "g", "B": "b", "C": "cyan", "D": "magenta"}

for i, ax in enumerate(axes):

    ax.hist(data[f"x{i + 1}"], bins="fd", density=True, color="gray")
    ax.plot(x_point, eval[f"p{i + 1}"].sum(axis=1))
    for j, state in enumerate(all_states[i]):
        print(j, state)
        ax.plot(x_point, eval[f"p{i + 1}"][:, j], color=colors[state])
    ax.format(title=f"Dataset {i + 1}")
pplt.show()


#%%

# Fitting only dataset two
model_ds2 = {Probability("p2"): model_dict[Probability("p2")]}

fit = Fit(model_ds2, x2=data["x2"])
result = fit.execute(guess=guess, minimizer=LikelihoodOptimizer, loss=LogSumLoss(sum_axis=1))

print(result.gof_qualifiers)

# Compare fit result with ground truth parameters
for k, v in result.parameters.items():
    print(f"{k:5}: {v:10.2}, ({gt[k]:10.2})")
