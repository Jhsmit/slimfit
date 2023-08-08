import proplot as pplt
from sympy import Matrix, exp
import numpy as np

from slimfit.numerical import GMM

from slimfit.fit import Fit
from slimfit.loss import LogSumLoss
from slimfit.markov import generate_transition_matrix, extract_states
from slimfit.minimizers import LikelihoodOptimizer
from slimfit.models import Model
from slimfit.objective import Hessian
from slimfit.operations import Mul
from slimfit.parameter import Parameters
from slimfit.symbols import clear_symbols, symbol_matrix, Symbol

# %%

arr = np.genfromtxt(r"C:\Users\jhsmi\pp\slimfit\examples\data\GMM_dynamics.txt")
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

# %%
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

# %%

parameters = Parameters.from_symbols(model.symbols, guess_values)

# %%
parameters.set("y0_A", lower_bound=0.0, upper_bound=1.0)  # mod? set_parameter ? modify?
parameters.set("y0_B", lower_bound=0.0, upper_bound=1.0, fixed=True)

parameters.set("k_A_B", lower_bound=1e-3, upper_bound=1e2)
parameters.set("k_B_A", lower_bound=1e-3, upper_bound=1e2)
parameters.set("k_B_C", lower_bound=1e-3, upper_bound=1e2)

# %%
# To calculate the likelihood for a measurement we need to sum the individual probabilities for all states
# Thus we need to define which axis this is in the model
STATE_AXIS = 1

# %%
fit = Fit(model, parameters, data, loss=LogSumLoss(sum_axis=STATE_AXIS))
result = fit.execute(
    minimizer=LikelihoodOptimizer,
    max_iter=200,
    verbose=True,
)


# %%
print(result)

# %%

num = 100
ti = np.linspace(0, 11, num=num, endpoint=True)
ei = np.linspace(-0.1, 1.1, num=num, endpoint=True)

grid = np.meshgrid(ti, ei, sparse=True)
grid

# %%
# since the `Mul` component of the model functions as a normal 'pyton' lazy multiplication,
# we can make use of numpy broadcasting to evaluate the model on a 100x100 datapoint grid

# %%
# timing: 1.83 ms
data_eval = {"t": ti.reshape(-1, 1), "e": ei.reshape(-1, 1)}
ans = model(**result.parameters, **data_eval)


# %%
# output shape is (N, N, 3, 1), we sum and squeeze to create the NxN grid
array = ans["p"].sum(axis=-2).squeeze()

# %%

fig, ax = pplt.subplots()
ax.contour(ti, ei, array.T, cmap="viridis", alpha=0.75)
ax.scatter(data["t"], data["e"], alpha=0.3, lw=0, color="k", zorder=-10)
ax.format(xlabel="t", ylabel="e", title="GMM dynamics")
pplt.show()
