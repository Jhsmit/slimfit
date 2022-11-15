import numpy as np
import proplot as pplt

from slimfit import Variable, Parameter, Model, Probability
from slimfit.fit import Fit
from slimfit.functions import gaussian_sympy

#%%

gt_params = {"mu": 2.4, "sigma": 0.7}

xdata = np.random.normal(gt_params["mu"], scale=gt_params["sigma"], size=500)
model = Model(
    {Probability("p"): gaussian_sympy(Variable("x"), Parameter("mu"), Parameter("sigma"))}
)


#%%

fit = Fit(model, x=xdata)
result = fit.execute()

#%%

data = {"x": np.linspace(0.0, 5.0, num=100)}

fig, ax = pplt.subplots()
ax.plot(data["x"], model(**data, **gt_params)["p"], color="r")
ax.plot(data["x"], model(**data, **result.parameters)["p"], linestyle="--", color="k")
ax.hist(xdata, bins="fd", density=True, color="grey")

pplt.show()
