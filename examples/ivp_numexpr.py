from typing import Optional

from scipy.integrate import solve_ivp
import numpy as np
import numpy.typing as npt
import proplot as pplt

from slimfit import Model, MatrixNumExpr
from slimfit.numerical import NumExprBase
from slimfit.fit import Fit
from slimfit.markov import generate_transition_matrix
from slimfit.numerical import to_numerical
from slimfit.symbols import SORT_KEY, FitSymbol, Parameter, Variable
from sympy import Matrix

#%%


class MarkovIVPNumExpr(NumExprBase):
    def __init__(
        self,
        t_var: Variable,
        trs_matrix: Matrix,
        y0: Matrix,
        domain: Optional[tuple[float, float]],
        **ivp_kwargs
    ):
        self.t_var = t_var
        self.trs_matrix = to_numerical(trs_matrix)
        self.y0 = to_numerical(y0)
        self.domain = domain

        ivp_defaults = {"method": "Radau"}
        self.ivp_defaults = ivp_defaults | ivp_kwargs

    @property
    def symbols(self) -> dict[str, FitSymbol]:
        symbols = self.trs_matrix.symbols | self.y0.symbols | {self.t_var.name: self.t_var}
        return {s.name: s for s in sorted(symbols, key=SORT_KEY)}

    def __call__(self, **kwargs):
        trs_matrix = self.trs_matrix(**kwargs)
        y0 = self.y0(**kwargs).squeeze()

        domain = self.domain or self.get_domain(**kwargs)
        sol = solve_ivp(
            self.grad_func,
            domain,
            y0=y0,
            t_eval=kwargs[self.t_var.name],
            args=(trs_matrix,),
            **self.ivp_defaults
        )

        return sol.y.T

    def get_domain(self, **kwargs) -> tuple[float, float]:
        dpts = kwargs[self.t_var.name]
        return dpts.min(), dpts.max()

    @staticmethod
    def grad_func(t, y, trs_matrix):
        return trs_matrix @ y

    def renew(self):
        self.trs_matrix.renew()
        self.y0.renew()


#%%

connectivity = ["A <-> B -> C"]
m = generate_transition_matrix(connectivity)
y0 = Matrix([[0.0, 1.0, 0.0]])
y0 = Matrix([[1.0, 0.0, 0.0]])

y0.shape

#%%

Parameter("k_A_B", vmin=1e-3, vmax=1e2)
Parameter("k_B_A", vmin=1e-3, vmax=1e2)
Parameter("k_B_C", vmin=1e-3, vmax=1e2)

ivp = MarkovIVPNumExpr(Variable("t"), m, y0, domain=(0.0, 11.0),)

ti = np.linspace(0, 11, num=250, endpoint=True)
gt_values = {
    "k_A_B": 5e-1,
    "k_B_A": 5e-2,
    "k_B_C": 2.5e-1,
}

arr = ivp(t=ti, **gt_values)

fig, ax = pplt.subplots()
for pop in arr.T:
    ax.plot(ti, pop)
pplt.show()

#%%
#
# ivp = IVPNumExpr(
#     Variable("t"), Parameter("freq"), Parameter("damp"), Parameter("y0"), domain=(0.0, 25.0),
# )
#
# #%%
#
# model = Model({Variable("y"): ivp})
# # Fix frequency at GT value to ensure fit converges
# Parameter("freq", value=0.2, fixed=True)
# Parameter("damp", vmax=0.5, value=0.0)
# Parameter("y0", value=-1.0)
# fit = Fit(model, y=data, t=t_eval)
# result = fit.execute()
#
# result.parameters, result.guess
#
# #%%
# fig, ax = pplt.subplots()
# ax.scatter(t_eval, data.T)
# ax.plot(t_eval, ivp(t=t_eval, **ivp.guess).T, color="r")
# ax.plot(t_eval, ivp(t=t_eval, **result.parameters).T, color="k")
# pplt.show()
