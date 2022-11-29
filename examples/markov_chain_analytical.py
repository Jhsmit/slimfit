from typing import Callable, Optional

from slimfit.callable import convert_callable
from slimfit.fit import Fit
from slimfit.markov import generate_transition_matrix, extract_states
from slimfit.symbols import Parameter, Variable, symbol_matrix
from slimfit.models import Model
import numpy as np
import proplot as pplt
from sympy import Matrix, lambdify, exp

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

#%%
states = extract_states(connectivity)

#%%
# model for markov chain
xt = exp(m * Variable("t"))

#%%
y0 = symbol_matrix(name="y0", shape=(3, 1), suffix=states, rand_init=True, norm=True)
model = Model({Variable("y"): xt @ y0})

#%%
# Generate data with 50 datapoints per population
num = 50
t = np.linspace(0, 11, num=num)

# Calling a matrix based model expands the dimensions of the matrix on the first axis to
# match the shape of input variables or parameters.
populations = model(t=t, **gt_values)["y"]

# add noise to populations
data = populations + np.random.normal(0, 0.05, size=num * 3).reshape(populations.shape)
data.shape  # shape of the data is (50, 3, 1)


#%%
# fit the model to the data
fit = Fit(model, y=data, t=t)
result = fit.execute()

#%%
# Compare fit result with ground truth parameters
for k, v in result.parameters.items():
    print(f"{k:5}: {v:10.2}, ({gt_values[k]:10.2})")
#%%

# compare input data to fitted population dynamics in a graph
color = ["#7FACFA", "#FA654D", "#8CAD36"]
cycle = pplt.Cycle(color=color)

t_eval = np.linspace(0, 11, 1000)
y_eval = model(**result.parameters, t=t_eval)["y"]

fig, ax = pplt.subplots()
c_iter = iter(cycle)
for pop in data.squeeze().T:
    ax.scatter(t, pop, **next(c_iter))

c_iter = iter(cycle)
for pop in y_eval.squeeze().T:
    ax.line(t_eval, pop, **next(c_iter))

pplt.show()
