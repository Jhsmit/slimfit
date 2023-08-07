# %% [markdown]
#
# This example assumes a measured spectrum which is a linear combination of known basis vectors
# (which are here modelled as Gaussians). The goal is to find the coefficients of the linear combinations.
#

# %%
import numpy as np

from slimfit import Model
from slimfit.fit import Fit
from slimfit.functions import gaussian_numpy

from slimfit.operations import MatMul
from slimfit.parameter import Parameters
from slimfit.symbols import symbol_matrix, Symbol

import proplot as pplt

# %% [markdown]
#
# Generating the basis vectors from Gaussians, create spectrum from given ground-truth coefficients.
#

# %%
# Generate the basis vectors, modelled as gaussian peaks
mu_vals = [1.1, 3.5, 7.2]
sigma_vals = [0.25, 0.1, 0.72]
wavenumber = np.linspace(0, 11, num=100)  # wavenumber x-axis
basis = np.stack(
    [gaussian_numpy(wavenumber, mu_i, sig_i) for mu_i, sig_i in zip(mu_vals, sigma_vals)]
).T
basis.shape
# %%
# Ground-truth coefficients we want to find
x_vals = np.array([0.3, 0.5, 0.2]).reshape(3, 1)  # unknowns

# Simulated measured spectrum given ground-truth coefficients and basis vectors
spectrum = basis @ np.array(x_vals).reshape(3, 1)
spectrum += np.random.normal(0, 0.1, size=spectrum.shape)  # add noise

spectrum.shape

# %% [markdown]
#
# The model describing the spectrum is of form $Ax = b$; where $A$ is the matrix of stacked basis vectors, $x$ is the
# coefficient vector we want to find and $b$ is the measured spectrum.
#
# We can define the model in two ways:
#
# **Option 1**: Create sympy Matrix with coefficients and multiply it with the array of coefficients.

# %%
# Create a sympy matrix with parameters are elements with shape (3, 1)
x = symbol_matrix(name="X", shape=(3, 1))
parameters = Parameters.from_symbols(x.free_symbols)
x, type(x)

# %%
# Matrix multiply basis matrix with parameter vector
m = basis @ x
# define the model, measured spectrum corresponds to Symbol('b')mod
model = Model({Symbol("b"): basis @ x})
type(model[Symbol("b")])

# %%
fit = Fit(model, parameters, data={"b": spectrum})
result = fit.execute()  # execution time 117 ms
result.parameters

# %% [markdown]
#
# This works but is performance-wise not desirable as the matrix in the model is shape (100, 1). Calling the model
# currently is implemented by lambdifying the matrix element-wise, this requires 100 lambda functions to be evaluated,
# and their result to be placed in an array through a for loop.
#
# **Option 2**: Evaluate the matrix multiplication lazily with `MatMul`
#
# %%
m = MatMul(basis, x)
model = Model({Symbol("b"): m})
type(model[Symbol("b")])

# %%
fit = Fit(model, parameters, data={"b": spectrum})
result = fit.execute()  # execution time: 15.2 ms

# %%
for i, j in np.ndindex(x_vals.shape):
    print(x_vals[i, j], result.parameters[f"X_{i}_{j}"])

# %% [markdown]
#
# The second option is 10 times faster as only a 3x1 matrix is evaluated through element-wise lambdification. The
# matrix multiplication step is done in numpy, and is fast.
#
# Plotting the results:
#
# %%

fig, ax = pplt.subplots()
ax.plot(wavenumber, spectrum, color="k")
ax.plot(wavenumber, model(**result.parameters)["b"], color="r", lw=1, alpha=0.75)
ax.format(xlabel="wavenumber", ylabel="intensity", title="Linear combination Fit")
pplt.show()
