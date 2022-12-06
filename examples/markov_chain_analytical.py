from typing import Callable, Optional

from slimfit.fit import Fit
from slimfit.markov import generate_transition_matrix, extract_states
from slimfit.parameter import Parameters
from slimfit.symbols import symbol_matrix, get_symbols
from slimfit.models import Model
import numpy as np
import proplot as pplt
from sympy import Matrix, lambdify, exp, Symbol

#%%
np.random.seed(43)

#%%
# Ground truth parameter values for generating data and fitting
gt_values = {
    "k_A_B": 1e0,
    "k_B_A": 5e-2,
    "k_B_C": 5e-1,
    "y0_A": 1.0,
    "y0_B": 0.0,
    "y0_C": 0.0,
}

#%%
# Generate markov chain transition rate matrix from state connectivity string(s)
connectivity = ["A <-> B -> C"]

m = generate_transition_matrix(connectivity)
states = extract_states(connectivity)

y0 = symbol_matrix(name="y0", shape=(3, 1), suffix=states)

# model for markov chain
xt = exp(m * Symbol("t"))
model = Model({Symbol("y"): xt @ y0})

#%%

rate_params = Parameters.from_symbols(m.free_symbols)
y0_params = Parameters.from_symbols(y0.free_symbols)
parameters = rate_params + y0_params

parameters

#%%
# Generate data with 50 datapoints per population
num = 50
xdata = {"t": np.linspace(0, 11, num=num)}

# Calling a matrix based model expands the dimensions of the matrix on the first axis to
# match the shape of input variables or parameters.
num_model = model.to_numerical()
populations = num_model(**gt_values, **xdata)["y"]
populations.shape


#%%

# # add noise to populations
ydata = {"y": populations + np.random.normal(0, 0.05, size=num * 3).reshape(populations.shape)}
ydata["y"].shape  # shape of the data is (50, 3, 1)


#%%
# fit the model to the data
fit = Fit(model, parameters, data={**xdata, **ydata})
result = fit.execute()

#%%
# Compare fit result with ground truth parameters
for k, v in result.parameters.items():
    print(f"{k:5}: {v:10.2}, ({gt_values[k]:10.2})")
#%%

# compare input data to fitted population dynamics in a graph
color = ["#7FACFA", "#FA654D", "#8CAD36"]
cycle = pplt.Cycle(color=color)

eval_data = {"t": np.linspace(0, 11, 1000)}
eval_model = model.to_numerical()
y_eval = eval_model(**result.parameters, **eval_data)["y"]

fig, ax = pplt.subplots()
c_iter = iter(cycle)
for pop in ydata["y"].squeeze().T:
    ax.scatter(xdata["t"], pop, **next(c_iter))

c_iter = iter(cycle)
for pop in y_eval.squeeze().T:
    ax.line(eval_data["t"], pop, **next(c_iter))
ax.format(xlabel="Time", ylabel="Population Fraction")
pplt.show()
