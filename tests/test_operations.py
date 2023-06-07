from slimfit.operations import Indexer
from slimfit.utils import format_indexer
import numpy as np


def test_indexer():
    arr = np.random.rand(10, 10, 10)

    indexer = np.index_exp[2:-1:8, 1:3, 5]
    idx = Indexer(arr, indexer)

    assert np.allclose(idx(), arr[indexer])
    assert format_indexer(indexer) == '[2:-1:8, 1:3, 5]'

    indexer = np.index_exp[:, :, :, np.newaxis]
    idx = Indexer(arr, indexer)
    assert idx().shape == (10, 10, 10, 1)
    assert np.allclose(idx(), arr[indexer])
    assert format_indexer(indexer) == '[:, :, :, None]'

    indexer = np.index_exp[..., 0]
    idx = Indexer(arr, indexer)
    assert np.allclose(idx(), arr[indexer])
    assert format_indexer(indexer) == '[..., 0]'



