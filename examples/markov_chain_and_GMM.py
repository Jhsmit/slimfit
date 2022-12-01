from slimfit import Parameter, Probability
from slimfit.numerical import GMM

from slimfit.fit import Fit
from slimfit.loss import LogSumLoss
from slimfit.markov import generate_transition_matrix, extract_states
from slimfit.minimizer import LikelihoodOptimizer
from slimfit.models import Model
from slimfit.operations import Mul
from slimfit.symbols import clear_symbols, Variable, parameter_matrix

from sympy import Matrix, exp
import numpy as np

#%%

arr = np.genfromtxt("data/GMM_dynamics.txt")
data = {"e": arr[:, 0], "t": arr[:, 1]}

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
    "y0_B": 0.2,
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
xt = exp(m * Variable("t"))
y0 = Matrix([[Parameter("y0_A"), Parameter("y0_B"), 1 - Parameter("y0_A") - Parameter("y0_B")]]).T

# Gaussian mixture model part
mu = parameter_matrix("mu", shape=(3, 1), suffix=states)
sigma = parameter_matrix("sigma", shape=(3, 1), suffix=states)
gmm = GMM(Variable("e"), mu=mu, sigma=sigma)

model = Model({Probability("p"): Mul(xt @ y0, gmm)})

#%%
# Future implementation needs constraints here
Parameter("y0_A", value=1.0, fixed=False, vmin=0.0, vmax=1.0)
Parameter("y0_B", value=0.0, fixed=True, vmin=0.0, vmax=1.0)
# y0_C is given by 1 - y0_A - y0_B

#%%
# Set bounds on rates
Parameter("k_A_B", vmin=1e-3, vmax=1e2)
Parameter("k_B_A", vmin=1e-3, vmax=1e2)
Parameter("k_B_C", vmin=1e-3, vmax=1e2)

#%%
# To calculate the likelihood for a measurement i we need to sum the individual probabilities for all states
# Thus we need to define which axis this is in the model
STATE_AXIS = 1

#%%

fit = Fit(model, **data)
result = fit.execute(
    guess=guess_values,
    minimizer=LikelihoodOptimizer,
    max_iter=100,
    verbose=True,
    loss=LogSumLoss(sum_axis=STATE_AXIS),
)

#%%

for k, v in result.parameters.items():
    print(f"{k:5}: {v:10.2}, ({gt_values[k]:10.2})")

#%%

data["t"].max()

#%%

num = 100
ti = np.linspace(0, 11, num=num, endpoint=True)
ei = np.linspace(-0.1, 1.1, num=num, endpoint=True)

grid = np.meshgrid(ti, ei, sparse=True)
grid

#%%
# since the `Mul` component of the model functions as a normal 'pyton' lazy multiplication,
# we can make use of numpy broadcasting to evaluate the model on a 100x100 datapoint grid

#%%
# timing: 3.45 ms
eval = model(t=ti.reshape(1, -1), e=ei.reshape(-1, 1), **result.parameters)  #

#%%
# output shape is (N, N, 3, 1), we sum and squeeze to create the NxN grid
array = eval["p"].sum(axis=-2).squeeze()

#%%
import proplot as pplt

fig, ax = pplt.subplots()
ax.contour(ti, ei, array, cmap="viridis")
ax.scatter(data["t"], data["e"], alpha=0.2, lw=0, color="k", zorder=-10)
ax.format(xlabel="t", ylabel="e")
fig.savefig("output/scatter_and_fit.png")
pplt.show()
