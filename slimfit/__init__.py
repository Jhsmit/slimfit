from slimfit.models import Model, Eval
from slimfit.numerical import MatrixNumExpr
from .base import NumExprBase
from slimfit.fit import Fit
from slimfit.symbols import Symbol, FitSymbol, symbol_matrix, clear_symbols
from slimfit.parameter import Parameters, Parameter
from slimfit.operations import Add, Mul, Sum

# placeholder version number
__version__ = "0.0.0"

# when we are on editable install from source, the _version file is present
# and we can get a version from there
try:
    from . import _version

    __version__ = _version.get_versions()["version"]
except ImportError:
    pass
