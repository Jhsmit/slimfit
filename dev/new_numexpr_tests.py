import itertools

from sympy import Symbol, Matrix, Expr, MatrixBase

from slimfit import Model
from slimfit.numerical import to_numerical, NumExpr, MatrixNumExpr, LambdaNumExpr, GMM
from slimfit.symbols import symbol_matrix, symbol_dict, get_symbols
from slimfit.parameter import Parameters, Parameter
import numpy as np

#%%


#%%
# model = Model({Symbol("y"): Symbol("a") * Symbol("x") + Symbol("b")})

expr = Symbol("a") * Symbol("x") + Symbol("b")

expr

data = {"x": np.arange(100).reshape(-1, 1)}
parameters = {
    "a": Parameter(Symbol("a"), guess=np.array([1, 2, 3]).reshape(1, -1)),
    "b": Parameter(Symbol("b"), guess=5.0),
}

num_expr = NumExpr(expr, parameters, data)
num_expr.shape

num_expr(a=np.random.rand(1, 3), b=5.0)

num_expr.shape

#%%


#%%


def func(x, a):
    return x ** 2 + a


data = {"x": np.arange(100)}

ld = LambdaNumExpr(
    func, [Symbol("a"), Symbol("x")], parameters={"a": Parameter(Symbol("a"), guess=3.0)}, data=data
)

assert ld.shape == (100,)

result = ld(a=2.0, **data)
assert np.allclose(result, data["x"] ** 2 + 2.0)


#%%
states = ["A", "B", "C"]
mu = symbol_matrix("mu", suffix=states)
sigma = symbol_matrix("sigma", suffix=states)
mu.shape

gmm = GMM(Symbol("x"), mu, sigma)
# symbols = get_symbols(gmm)
parameters = Parameters.from_symbols(gmm.symbols, "mu_A mu_B mu_C sigma_A sigma_B sigma_C")

gt = {
    "mu_A": 0.23,
    "mu_B": 0.55,
    "mu_C": 0.92,
    "sigma_A": 0.1,
    "sigma_B": 0.1,
    "sigma_C": 0.1,
    "c_A": 0.22,
    "c_B": 0.53,
    "c_C": 0.25,
}

type(gmm.mu), type(gmm.sigma)


#%%
num_gmm = gmm.to_numerical(parameters, data)
print(num_gmm["x"].lambdified.__doc__)
#%%

num_gmm(**gt).shape
#%%

data = {"x": np.linspace(-0.2, 1.2, num=25).reshape(1, -1)}
model = Model({Symbol("y"): GMM(Symbol("x"), mu, sigma)})
num_model = model.to_numerical(parameters, data)

num_model.data


num_model(**gt)["y"].shape
