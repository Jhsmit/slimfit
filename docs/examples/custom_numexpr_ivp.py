# %% [markdown]
#
# This example shows how to use `CompositeExpr` to create a custom numerical expression which can
# be used in slimfit fitting.
#
# In this particular example we fit data to a dampened harmonic oscillator, where the time-evolution
# of the system is solved by `scipy.integrate.solve_ivp`.

from __future__ import annotations

import numpy as np
import proplot as pplt
from scipy.integrate import solve_ivp
from sympy import Symbol, Expr, symbols

from slimfit import Model
from slimfit.fit import Fit
from slimfit.numerical import NumExpr, to_numerical
from slimfit.base import CompositeExpr


# %%

# %% [markdown]
#
# Generate the GT data to fit the damped harmonic oscillator to, and add some noise.


def ode(x, y):
    return np.sin(2 * np.pi * 0.2 * x) * np.exp(-0.1 * x)


num = 100
t_eval = np.linspace(0.0, 25, num=num, endpoint=True)
sol = solve_ivp(ode, (0.0, 25), np.array([-1]), t_eval=t_eval)

ydata = sol.y + np.random.normal(0, 0.05, size=num)
data = {"y": ydata, "t": t_eval}

# %%

# %% [markdown]
#
# `CompositeExpr` can be subclassed to create a custom numerical expression. The subclass must
# implement the `__call__` method, which returns a (dictionary of) the numerical values of the
# expression. In this example, we use `solve_ivp` to solve the ODE, and return the solution at the
# specified time points.
#
# Because the `__init__` method takes an additional `domain` argument, the `to_numerical` method
# must also be implemented correctly.


class IVPNumExpr(CompositeExpr):
    def __init__(
        self,
        t: Symbol | NumExpr | Expr,
        freq: Symbol | NumExpr | Expr,
        damping: Symbol | NumExpr | Expr,
        y0: Symbol | NumExpr | Expr,
        domain: tuple[float, float],
    ):
        expr = {"t": t, "freq": freq, "damping": damping, "y0": y0}
        self.domain = domain
        super().__init__(expr)

    def __call__(self, *args, **kwargs) -> np.ndarray:
        result = super().__call__(**kwargs)

        sol = solve_ivp(
            self.grad_func,
            self.domain,
            np.array([result["y0"]]),
            t_eval=result["t"],
            args=(result["freq"], result["damping"]),
        )

        return sol.y

    def to_numerical(self):
        num_expr = {k: to_numerical(expr) for k, expr in self.items()}
        instance = IVPNumExpr(**num_expr, domain=self.domain)

        return instance

    @staticmethod
    def grad_func(x, y, freq, damping):
        return np.sin(2 * np.pi * freq * x) * np.exp(-damping * x)


# %%

# %% [markdown]
#
# The resulting class can now be used in slimfit fitting, taking any symbol or expr as arguments for
# the args `t, f, d, y0`, or it can be embedded in a larger model.

t, f, d, y0, y = symbols("t f d y0 y")
ivp = IVPNumExpr(t, f, d, y0, domain=(0.0, 25.0))

model = Model({y: ivp})

# Fix frequency at GT value to ensure fit converges
guess = {"f": 0.2, "d": 0.5, "y0": -1.0}
parameters = model.define_parameters(guess).replace("f", fixed=True)

fit = Fit(model, parameters, data)
result = fit.execute()

print(result.parameters)

# %%

fig, ax = pplt.subplots()
ax.scatter(t_eval, ydata.flatten())
ax.plot(t_eval, ivp(t=t_eval, **parameters.guess).T, color="r")
ax.plot(t_eval, ivp(t=t_eval, **result.parameters).T, color="k")
pplt.show()
