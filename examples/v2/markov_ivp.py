"""
Same problem as in 'markov_chain_analytical', but now solved numerically rather than with
matrix exponentiation.

The numerical IVP solver is useful for larger models where matrix exponentiation is no longer
feasible.

"""

from __future__ import annotations

import numpy as np
import proplot as pplt
from sympy import Symbol

from slimfit.markov import extract_states, generate_transition_matrix
from slimfit.symbols import symbol_matrix
from slimfit.v2.composite_expr import MarkovIVP
from slimfit.v2.loss import MSELoss
from slimfit.v2.minimize import Minimize
from slimfit.v2.model import Model
from slimfit.v2.parameter import Parameters

# %%
np.random.seed(43)

# %%
# Ground truth parameter values for generating data and fitting
gt_values = {
    "k_A_B": 1e0,
    "k_B_A": 5e-2,
    "k_B_C": 5e-1,
    "y0_A": 1.0,
    "y0_B": 0.0,
    "y0_C": 0.0,
}

# %%
# Generate markov chain transition rate matrix from state connectivity string(s)
connectivity = ["A <-> B -> C"]
m = generate_transition_matrix(connectivity)


# %%
states = extract_states(connectivity)
y0 = symbol_matrix(name="y0", shape=(3, 1), suffix=states)
model = Model({Symbol("y"): MarkovIVP(Symbol("t"), m, y0)})

rate_params = Parameters.from_symbols(m.free_symbols)
y0_params = Parameters.from_symbols(y0.free_symbols)

parameters = rate_params + y0_params
# %%
num = 50
xdata = {"t": np.linspace(0, 11, num=num)}

# Calling a matrix based model expands the dimensions of the matrix on the first axis to
# match the shape of input variables or parameters.
populations = model(**gt_values, **xdata)["y"]
populations.shape

# %%
# add noise to populations
ydata = {"y": populations + np.random.normal(0, 0.05, size=num * 3).reshape(populations.shape)}
ydata["y"].shape  # shape of the data is (3, 50)

# %%
loss = MSELoss(model, ydata)
objective = Minimize(loss, parameters, xdata)
result = objective.fit()

# %%
# Compare fit result with ground truth parameters
for k, v in result.parameters.items():
    print(f"{k:5}: {v:10.2}, ({gt_values[k]:10.2})")

# %%
color = ["#7FACFA", "#FA654D", "#8CAD36"]
cycle = pplt.Cycle(color=color)

eval_data = {"t": np.linspace(0, 11, 1000)}
y_eval = model(**result.parameters, **eval_data)["y"]

fig, ax = pplt.subplots()
ax.scatter(xdata["t"], ydata["y"].T, cycle=cycle)
ax.line(eval_data["t"], y_eval.T, cycle=cycle)
ax.format(xlabel="Time", ylabel="Population Fraction")
pplt.show()
