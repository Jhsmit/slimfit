from slimfit.models import Model
from slimfit.numerical import NumExprBase, MatrixNumExpr
from slimfit.fit import Fit
from slimfit.symbols import Symbol, FitSymbol
from slimfit.parameter import Parameters, Parameter

# placeholder version number
__version__ = "0.0.0"

# when we are on editable install from source, the _version file is present
# and we can get a version from there
try:
    from . import _version
    __version__ = _version.get_versions()["version"]
except ImportError:
    pass
