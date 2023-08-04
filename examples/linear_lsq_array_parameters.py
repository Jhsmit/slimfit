# %%

from slimfit.fit import Fit
from slimfit.parameter import Parameters
from slimfit.symbols import Symbol
from slimfit.models import Model

import numpy as np
import proplot as pplt

# %%
model = Model({Symbol("y"): Symbol("a") * Symbol("x") + Symbol("b")})

# %%
# Generate Ground-Truth data
gt = {"a": 0.15, "b": 2.5}

xdata = np.linspace(0, 11, num=100)
ydata = gt["a"] * xdata + gt["b"]

noise = np.random.normal(0, scale=ydata / 10.0 + 0.2)
ydata += noise

DATA = {"x": xdata, "y": ydata}

# %%
# compare to numpy polyfit
np.polyfit(xdata, ydata, deg=1)

# %%

parameters = Parameters.from_symbols(model.symbols, "a b")
fit = Fit(model, parameters=parameters, data=DATA)
result = fit.execute()


# %%
result.eval_hessian()
print(result)

# %%
fig, ax = pplt.subplots()
ax.scatter(DATA["x"], DATA["y"])
ax.plot(DATA["x"], model.numerical(**result.parameters, **DATA)["y"], color="r")
pplt.show()
