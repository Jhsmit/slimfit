from slimfit import Fit, Model, Parameter
from sympy import symbols
import numpy as np
import proplot as pplt

from slimfit.objective import Hessian

a, b, x, y = symbols("a b x y")

model = Model({y: a * x + b})
parameters = [
    Parameter(a, guess=1.0),
    Parameter(b, guess=3.0),
]
# %%
# generate ground-truth data
gt = {a: 0.5, b: 2.5}
xdata = np.linspace(0, 11, num=100)
ydata = gt[a] * xdata + gt[b]

# add noise
np.random.seed(43)
noise = np.random.normal(0, scale=ydata / 10.0 + 0.2)
ydata += noise
DATA = {"x": xdata, "y": ydata}

fit = Fit(model, parameters=parameters, data=DATA)
result = fit.execute()
print(result)


# %%
fig, ax = pplt.subplots()
ax.scatter(xdata, ydata)
ax.plot(DATA["x"], model(**result.parameters, **DATA)["y"], color="r")
ax.format(xlabel="x", ylabel="y")
pplt.show()
