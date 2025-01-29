# %%

import numpy as np

from slimfit.v2.loss import MSELoss
from slimfit.v2.minimize import Minimize
from slimfit.v2.model import Model
from slimfit.v2.symbol import Symbols

# %%

s = Symbols("x y a b")
model = Model({s.y: s.a * s.x + s.b})  # type: ignore

# Generate Ground-Truth data
np.random.seed(43)
gt = {"a": 0.15, "b": 2.5}

xdata = np.linspace(0, 11, num=100)
ydata = gt["a"] * xdata + gt["b"]

noise = np.random.normal(0, scale=ydata / 10.0 + 0.2)
ydata += noise
# %%
parameters = model.define_parameters("a b")
parameters
# %%

loss = MSELoss(model, dict(y=ydata))
objective = Minimize(loss, parameters, dict(x=xdata))
result = objective.fit()
result

# %%
# %%
# compare to numpy polyfit
np.polyfit(xdata, ydata, deg=1)


# %%
import proplot as pplt

fig, ax = pplt.subplots()
ax.scatter(xdata, ydata)
ax.plot(xdata, model(**result.parameters, x=xdata)["y"], color="r")
pplt.show()
# %%
