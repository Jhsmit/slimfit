from slimfit.fit import Fit
from slimfit.parameter import Parameters
from slimfit.symbols import Symbol
from slimfit.models import Model

import numpy as np
import proplot as pplt

#%%
model = Model({Symbol("y"): Symbol("a") * Symbol("x") + Symbol("b")})

#%%
# Generate Ground-Truth data
gt = {"a": 0.5, "b": 2.5}

xdata = np.linspace(0, 11, num=100)
ydata = gt["a"] * xdata + gt["b"]

noise = np.random.normal(0, scale=ydata / 10.0 + 0.2)
ydata += noise

DATA = {"x": xdata, "y": ydata}

#%%

parameters = Parameters.from_symbols(model.symbols, "a b")

fit = Fit(model, parameters=parameters, data=DATA)
#%%

#%%

fit.numerical_model.parameters

#%%
result = fit.execute()

#%%
fig, ax = pplt.subplots()
ax.scatter(DATA["x"], DATA["y"])
ax.plot(DATA["x"], model.to_numerical(parameters, DATA)(**result.parameters)["y"], color="r")
pplt.show()
