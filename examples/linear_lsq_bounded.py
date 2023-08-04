# %%

from slimfit.fit import Fit
from slimfit.minimizers import ScipyMinimizer
from slimfit.parameter import Parameters, Parameter
from slimfit.symbols import Symbol
from slimfit.models import Model

import numpy as np
import proplot as pplt

# %%
np.random.seed(43)

gt = {"a": 0.5, "b": np.array([1.0, 2.0, 3.0]).reshape(3, 1)}

xdata = np.linspace(0, 11, num=100)
ydata = gt["a"] * xdata + gt["b"]

noise = np.random.normal(0, scale=ydata / 10.0 + 0.5)
ydata += noise

data = {"x": xdata, "y": ydata}

# %%

model = Model({Symbol("y"): Symbol("a") * Symbol("x") + Symbol("b")})
symbols = {s.name: s for s in model.symbols}
parameters = Parameters(
    [
        Parameter(symbols["a"], guess=1.0),
        Parameter(symbols["b"], guess=np.ones((3, 1)), lower_bound=0.0),
    ]
)

fit = Fit(model, parameters, data)
res = fit.execute()

# %%
print(res.parameters)
print(res.stdev)

# %%
x_eval = np.linspace(0, 11, num=100)
fig, ax = pplt.subplots()
ax.scatter(xdata, ydata.T, cycle="default", s=7.5)
ax.plot(x_eval, model(**res.parameters, x=x_eval)["y"].T)
ax.format(xlabel="x", ylabel="y")
pplt.show()

# %%
