import numpy as np

from slimfit import Model
from slimfit.fit import Fit
from slimfit.functions import gaussian_numpy

from slimfit.operations import MatMul
from slimfit.symbols import parameter_matrix, Variable

import proplot as pplt


"""
In this example we have a measured spectrum which consist of a linear combination of known basis vectors and 
want to find the coefficients of these linear combinations. 
"""

#%%

# Generate the basis vectors, modelled as gaussian peaks
mu_vals = [1.1, 3.5, 7.2]
sigma_vals = [0.25, 0.1, 0.72]
wavenumber = np.linspace(0, 11, num=100)  # wavenumber x-axis
basis = np.stack(
    [gaussian_numpy(wavenumber, mu_i, sig_i) for mu_i, sig_i in zip(mu_vals, sigma_vals)]
).T

#%%

# Ground-truth coefficients we want to find
x_vals = np.array([0.3, 0.5, 0.2]).reshape(3, 1)  # unknowns

# Simulated measured spectrum given ground-truth coefficients and basis vectors
spectrum = basis @ np.array(x_vals).reshape(3, 1)

#%%

"""
The model describing the spectrum is of from Ax = b; where A is our matrix of stacked basis vectors, x is the 
coefficient vector we want to find and b is the measured spectrum.

We can define the model in two ways: 
"""

#%%
"""
Option 1: Create sympy Matrix with coefficients and multiply it with the array of coefficients
"""
# Create a sympy matrix with parameters are elements with shape (3, 1)
x = parameter_matrix(name="X", shape=(3, 1))
x

#%%

m = basis @ x  # Matirx multiply basis matrix with parameter vector
model = Model(
    {Variable("b"): basis @ x}
)  # define the model, measured spectrum corresponds to Variable('b')
m_callable = model.model_dict[Variable("b")]
m_callable.shape

#%%
fit = Fit(model, b=spectrum)
result = fit.execute()  # executiong time 117 ms
result.parameters

#%%

"""
This works but is performance-wise not desirable as the MatrixCallable in the model is shape (100, 1) and calling it
requires evalulating one lambdified function per matrix element.
"""

#%%

"""
Option 2: Create a (3x1) Matrix and evaluate the matrix multiplication lazily with `MatMul`
"""

#%%
m = MatMul(basis, x)
model = Model({Variable("b"): m})
m_callable = model.model_dict[Variable("b")]
m_callable  # = MatMul object
#%%

fit = Fit(model, b=spectrum)
result = fit.execute()  # execution time: 13.3 ms

#%%

result.parameters

for i, j in np.ndindex(x_vals.shape):
    print(x_vals[i, j], result.parameters[f"X_{i}_{j}"])


#%%
# plot the results
fig, ax = pplt.subplots()
ax.plot(wavenumber, spectrum, color="r")
ax.plot(wavenumber, model(**result.parameters)["b"], color="k", linestyle="--")
pplt.show()