from slimfit.numerical import GMM

from slimfit.fit import Fit
from slimfit.loss import LogSumLoss
from slimfit.markov import generate_transition_matrix, extract_states
from slimfit.minimizer import LikelihoodOptimizer
from slimfit.models import Model
from slimfit.operations import Mul
from slimfit.parameter import Parameters, Parameter
from slimfit.symbols import clear_symbols, symbol_matrix, Symbol

# from slimfit.symbols import FitSymbol as Symbol

from sympy import Matrix, exp
import numpy as np

#%%

arr = np.genfromtxt("data/GMM_dynamics.txt")
data = {"e": arr[:, 0].reshape(-1, 1), "t": arr[:, 1]}

gt_values = {
    "k_A_B": 5e-1,
    "k_B_A": 5e-2,
    "k_B_C": 2.5e-1,
    "y0_A": 1.0,
    "y0_B": 0.0,
    "mu_A": 0.82,
    "mu_B": 0.13,
    "mu_C": 0.55,
    "sigma_A": 0.095,
    "sigma_B": 0.12,
    "sigma_C": 0.08,
}

guess_values = {
    "k_A_B": 1e-1,
    "k_B_A": 1e-1,
    "k_B_C": 1e-1,
    "y0_A": 0.6,
    "y0_B": 0.0,
    "mu_A": 0.7,
    "mu_B": 0.05,
    "mu_C": 0.4,
    "sigma_A": 0.1,
    "sigma_B": 0.2,
    "sigma_C": 0.1,
}

#%%
clear_symbols()

connectivity = ["A <-> B -> C"]
m = generate_transition_matrix(connectivity)
states = extract_states(connectivity)

# Temporal part
xt = exp(m * Symbol("t"))
y0 = Matrix([[Symbol("y0_A"), Symbol("y0_B"), 1 - Symbol("y0_A") - Symbol("y0_B")]]).T

# Gaussian mixture model part
mu = symbol_matrix("mu", shape=(1, 3), suffix=states)
sigma = symbol_matrix("sigma", shape=(1, 3), suffix=states)
gmm = GMM(Symbol("e"), mu=mu, sigma=sigma)

model = Model({Symbol("p"): Mul(xt @ y0, gmm)})

#%%

parameters = Parameters.from_symbols(model.symbols, guess_values)

#%%

# Future implementation needs constraints here
parameters["y0_A"].lower_bound = 0.0
parameters["y0_A"].upper_bound = 1.0

parameters["y0_B"].lower_bound = 0.0
parameters["y0_B"].upper_bound = 1.0
parameters["y0_B"].fixed = True

#%%
# Set bounds on rates
parameters["k_A_B"].lower_bound = 1e-3
parameters["k_A_B"].upper_bound = 1e2

parameters["k_B_A"].lower_bound = 1e-3
parameters["k_B_A"].upper_bound = 1e2

parameters["k_B_C"].lower_bound = 1e-3
parameters["k_B_C"].upper_bound = 1e2

#%%
# To calculate the likelihood for a measurement we need to sum the individual probabilities for all states
# Thus we need to define which axis this is in the model
STATE_AXIS = 1

#%%
fit = Fit(model, parameters, data, loss=LogSumLoss(sum_axis=STATE_AXIS))
result = fit.execute(minimizer=LikelihoodOptimizer, max_iter=200, verbose=True,)

#%%
for k, v in result.parameters.items():
    print(f"{k:5}: {v:10.2}, ({gt_values[k]:10.2})")

#%%

num = 100
ti = np.linspace(0, 11, num=num, endpoint=True)
ei = np.linspace(-0.1, 1.1, num=num, endpoint=True)

grid = np.meshgrid(ti, ei, sparse=True)
grid
#
#%%
# since the `Mul` component of the model functions as a normal 'pyton' lazy multiplication,
# we can make use of numpy broadcasting to evaluate the model on a 100x100 datapoint grid

#%%
# timing: 2.33 ms
data_eval = {"t": ti.reshape(-1, 1), "e": ei.reshape(-1, 1)}
num_model = model.to_numerical(parameters, data_eval)
ans = num_model(**result.parameters)
ans["p"].shape

#%%
# output shape is (N, N, 3, 1), we sum and squeeze to create the NxN grid
array = ans["p"].sum(axis=-2).squeeze()

#%%
import proplot as pplt

fig, ax = pplt.subplots()
ax.contour(ti, ei, array.T, cmap="viridis")
ax.scatter(data["t"], data["e"], alpha=0.2, lw=0, color="k", zorder=-10)
ax.format(xlabel="t", ylabel="e")
fig.savefig("output/scatter_and_fit.png")
pplt.show()
