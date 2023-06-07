from sympy import Symbol

from slimfit.numerical import NumExpr
from slimfit.operations import Indexer, Add
from slimfit.utils import format_indexer
import numpy as np


def test_indexer():
    arr = np.random.rand(10, 10, 10)

    indexer = np.index_exp[2:-1:8, 1:3, 5]
    idx = Indexer(arr, indexer)

    assert np.allclose(idx(), arr[indexer])
    assert format_indexer(indexer) == '[2:-1:8, 1:3, 5]'

    expr = Symbol('a')
    assert Indexer(expr, indexer).__repr__() == 'a[2:-1:8, 1:3, 5]'

    indexer = np.index_exp[:, :, :, np.newaxis]
    idx = Indexer(arr, indexer)
    assert idx().shape == (10, 10, 10, 1)
    assert np.allclose(idx(), arr[indexer])
    assert format_indexer(indexer) == '[:, :, :, None]'

    indexer = np.index_exp[..., 0]
    idx = Indexer(arr, indexer)
    assert np.allclose(idx(), arr[indexer])
    assert format_indexer(indexer) == '[..., 0]'

    a = np.random.rand(10, 1)
    b = np.random.rand(10, 1)

    assert np.allclose(Add(a, b)[3:5](), (a + b)[3:5])


def test_transpose():
    a = np.random.rand(10, 1)
    b = np.random.rand(10, 1)
    assert np.allclose(Add(a, b).T(), (a + b).T)

    expr = NumExpr(Symbol('a') + Symbol('b'))

    assert np.allclose(expr.T(a=a, b=b), (a + b).T)


#%%





