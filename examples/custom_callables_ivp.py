from scipy.integrate import solve_ivp
import numpy as np
import numpy.typing as npt
import proplot as pplt

from slimfit import Model
from slimfit.callable import NumExprBase
from slimfit.fit import Fit
from slimfit.symbols import SORT_KEY, FitSymbol, Parameter, Variable


#%%


def ode(x, y):
    return np.sin(2 * np.pi * 0.2 * x) * np.exp(-0.1 * x)


num = 100
t_eval = np.linspace(0.0, 25, num=num, endpoint=True)
sol = solve_ivp(ode, (0.0, 25), np.array([-1]), t_eval=t_eval)

#%%
data = sol.y + np.random.normal(0, 0.05, size=num)

#%%


class IVPNumExpr(NumExprBase):
    def __init__(
        self,
        t_var: Variable,
        freq: Parameter,
        damping: Parameter,
        y0: Parameter,
        domain: tuple[float, float],
    ):
        self.t_var = t_var
        self.freq = freq
        self.damping = damping
        self.y0 = y0
        self.domain = domain

    @property
    def symbols(self) -> dict[str, FitSymbol]:
        symbols = [self.t_var, self.freq, self.damping, self.y0]
        return {s.name: s for s in sorted(symbols, key=SORT_KEY)}

    def __call__(self, *args, **kwargs):
        sol = solve_ivp(
            self.grad_func,
            self.domain,
            np.array([self.get_val(kwargs, self.y0)]),
            t_eval=self.get_val(kwargs, self.t_var),
            args=(self.get_val(kwargs, self.freq), self.get_val(kwargs, self.damping)),
        )

        return sol.y

    def get_val(self, kwargs, symbol) -> npt.ArrayLike:
        if getattr(symbol, "fixed", False):
            return symbol.value
        else:
            return kwargs[symbol.name]

    @staticmethod
    def grad_func(x, y, freq, damping):
        return np.sin(2 * np.pi * freq * x) * np.exp(-damping * x)

    def renew(self):
        ...


ivp = IVPNumExpr(
    Variable("t"), Parameter("freq"), Parameter("damp"), Parameter("y0"), domain=(0.0, 25.0),
)

#%%

model = Model({Variable("y"): ivp})
# Fix frequency at GT value to ensure fit converges
Parameter("freq", value=0.2, fixed=True)
Parameter("damp", vmax=0.5, value=0.0)
Parameter("y0", value=-1.0)
fit = Fit(model, y=data, t=t_eval)
result = fit.execute()

result.parameters, result.guess

#%%
fig, ax = pplt.subplots()
ax.scatter(t_eval, data.T)
ax.plot(t_eval, ivp(t=t_eval, **ivp.guess).T, color="r")
ax.plot(t_eval, ivp(t=t_eval, **result.parameters).T, color="k")
pplt.show()
