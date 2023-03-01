# %% [markdown]
#
# Likelihood fitting of normally distributed samples.
#

#%%
import numpy as np
import proplot as pplt

from slimfit.symbols import Symbol
from slimfit.models import Model
from slimfit.fit import Fit
from slimfit.functions import gaussian_sympy
from slimfit.loss import LogLoss
from slimfit.parameter import Parameters

#%%

gt_params = {"mu": 2.4, "sigma": 0.7}

xdata = np.random.normal(gt_params["mu"], scale=gt_params["sigma"], size=500)
model = Model({Symbol("p"): gaussian_sympy(Symbol("x"), Symbol("mu"), Symbol("sigma"))})

#%%
parameters = Parameters.from_symbols(model.symbols, "mu sigma")
#%%
fit = Fit(model, parameters, data={"x": xdata}, loss=LogLoss())

# execution time: 12.5ms
result = fit.execute()

#%%
data = {"x": np.linspace(0.0, 5.0, num=100)}

fig, ax = pplt.subplots()
ax.plot(data["x"], model(**gt_params, **data)["p"], color="r")
ax.plot(data["x"], model(**result.parameters, **data)["p"], linestyle="--", color="k")
ax.hist(xdata, bins="fd", density=True, color="grey")
ax.format(xlabel="x", ylabel="p(x)", title="Gaussian Likelihood Fit")
pplt.show()
