from slimfit.fit import Fit
from slimfit.symbols import Parameter, Variable
from slimfit.models import Model

import numpy as np
import proplot as pplt

#%%
model = Model({Variable("y"): Parameter("a") * Variable("x") + Parameter("b")})

#%%

gt = {"a": 0.5, "b": 2.5}

xdata = np.linspace(0, 11, num=100)
ydata = gt["a"] * xdata + gt["b"]

noise = np.random.normal(0, scale=ydata / 10.0 + 0.2)
ydata += noise

data = {"x": xdata, "y": ydata}

#%%

fit = Fit(model, **data)
result = fit.execute()

#%%

fig, ax = pplt.subplots()
ax.scatter(fit.independent_data["x"], fit.dependent_data["y"])
ax.plot(data["x"], model(**fit.independent_data, **result.parameters)["y"], color="r")
pplt.show()

#%%

print(result.gof_qualifiers)
