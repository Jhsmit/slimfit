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
gt = {"a": 0.15, "b": 2.5}

xdata = np.linspace(0, 11, num=100)
ydata = gt["a"] * xdata + gt["b"]

noise = np.random.normal(0, scale=ydata / 10.0 + 0.2)
ydata += noise

DATA = {"x": xdata, "y": ydata}

#%%

np.polyfit(xdata, ydata, deg=1)

#%%

parameters = Parameters.from_symbols(model.symbols, "a b")
fit = Fit(model, parameters=parameters, data=DATA)

#%%
result = fit.execute()
result.parameters
#
#%%
fig, ax = pplt.subplots()
ax.scatter(DATA["x"], DATA["y"])
ax.plot(DATA["x"], model.numerical(**result.parameters, **DATA)["y"], color="r")
pplt.show()
#
#


#%%

from sympy import sin
global_model = Model({
    Symbol("y"): Symbol("a") * Symbol("x") + Symbol("b"),
    Symbol("z"): sin(Symbol('f')*Symbol('t') + Symbol('phi')) + Symbol('b'),
})
global_model

#%%
# Symbol("t") ** Symbol("c") + Symbol("b"),

# Note that you must not lose the cache

#%%

freq = 1.2
phase = np.array([0.24*np.pi, 0.8*np.pi, 1.3*np.pi]).reshape(3, 1)
tdata = np.linspace(0, 11, num=25) + np.random.normal(size=25, scale=0.25)
zdata = np.sin(freq*tdata + phase) + gt["b"]
#%%

GLOBAL_DATA = {"t": tdata, "z": zdata, "f": 1.2, **DATA}
GLOBAL_DATA

#%%

guess = {
    'a': 1.,
    'b': 1.,
    'phi': np.ones((3, 1))
}

global_parameters = Parameters.from_symbols(global_model.symbols, guess)
parameters

#%%

fit = Fit(global_model, parameters=global_parameters, data=GLOBAL_DATA)
global_result = fit.execute()

result.parameters


#%%

# TODO: FREQUENCY FITTING IS TOO HARD ?

fix, axes = pplt.subplots(ncols=2, share=False)
axes[0].scatter(xdata, ydata)
axes[0].axline((0, result.parameters["b"]), slope=result.parameters["a"], color="r")
axes[0].axline((0, global_result.parameters["b"]), slope=global_result.parameters["a"], color="k")
axes[0].format(xlabel='x', ylabel='y')
tvec = np.linspace(0, 11, num=250)
#z_eval = global_model.numerical['z'](t=tvec, f=freq, phi=phase, b=gt['b'])

z_eval = global_model.numerical['z'](**global_result.parameters, t=tvec, f=1.2)

#ax.scatter(xdata, ydata, color='r')
axes[1].plot(tvec, z_eval.T, alpha=0.5, cycle='default')
axes[1].scatter(tdata, zdata.T)
axes[1].format(xlabel='t', ylabel='z')
pplt.show()

#%%

global_result.parameters['b']
#%%

fig, ax = pplt.subplots()
ax.plot(tdata, zdata.T)
pplt.show()


#%%
eval = global_model.numerical['z'](t=tvec, f=freq, phi=phase, b=gt['b'])
eval.shape
#%%
cycle = pplt.Cycle('qual1')
next(iter(cycle))
