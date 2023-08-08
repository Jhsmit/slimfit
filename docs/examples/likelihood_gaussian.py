# %% [markdown]
#
# Likelihood fitting of normally distributed samples.
#

# %%
import numpy as np
import proplot as pplt

from slimfit.symbols import Symbol
from slimfit.models import Model
from slimfit.fit import Fit
from slimfit.functions import gaussian_sympy
from slimfit.loss import LogLoss
from slimfit.parameter import Parameters

# %%

gt_params = {"mu": 2.4, "sigma": 0.7}

xdata = np.random.normal(gt_params["mu"], scale=gt_params["sigma"], size=500)
model = Model({Symbol("p"): gaussian_sympy(Symbol("x"), Symbol("mu"), Symbol("sigma"))})

# %%
parameters = Parameters.from_symbols(model.symbols, "mu sigma")
# %%
fit = Fit(model, parameters, data={"x": xdata}, loss=LogLoss())

# execution time: 12.5ms
likelihood_result = fit.execute()

# %%

hist, edges = np.histogram(xdata, bins="fd", density=True)
centers = (edges[:-1] + edges[1:]) / 2.0
centers, edges

# %%

fig, ax = pplt.subplots()
ax.scatter(centers, hist, color="r")
ax.hist(xdata, bins="fd", density=True, color="gray")
pplt.show()

# %%
fit = Fit(model, parameters, data={"x": centers, "p": hist})
binned_lsq_result = fit.execute()
print(binned_lsq_result)
# %%
binned_lsq_result.guess
# %%
print(likelihood_result.hessian)
print(binned_lsq_result.hessian)
# %%
data = {"x": np.linspace(0.0, 5.0, num=100)}

fig, ax = pplt.subplots()
ax.plot(data["x"], model(**gt_params, **data)["p"], color="r")
ax.plot(
    data["x"],
    model(**likelihood_result.parameters, **data)["p"],
    linestyle="--",
    color="k",
    label="likelihood",
)
ax.plot(
    data["x"],
    model(**binned_lsq_result.parameters, **data)["p"],
    linestyle="--",
    color="b",
    label="lsq",
)

ax.hist(xdata, bins="fd", density=True, color="grey", zorder=-1)
ax.scatter(centers, hist, color="b", marker="x")
ax.format(xlabel="x", ylabel="p(x)", title="Gaussian Fit")
ax.legend(loc="t")
pplt.show()
