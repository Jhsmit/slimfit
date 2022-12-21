# SlimFit

**SymFit's little brother**

Documentation: https://jhsmit.github.io/slimfit/

This project is inspired by [SymFit](https://github.com/tBuLi/symfit) and is functional, to some degree, but in currently in BETA

* Free software: MIT license


## Aims

* Inspiration for a potential SymFit 2.0
* Expectation-Maximization likelihood maximization

## Quick Start

```python
from sympy import symbols
from slimfit import Model, Fit, Parameter
import numpy as np

y, a, x, b = symbols('y a x b')

model = Model({y: a*x + b})
parameters = [
    Parameter(a, guess=2.5),
    Parameter(b, guess=1, lower_bound=0.)
]

xdata = np.linspace(0, 11, 25)
ydata = 0.5*xdata + 2.5
ydata += np.random.normal(0, scale= ydata / 10.0 + 0.2)
data = {'x': xdata, 'y': ydata}

fit = Fit(model, parameters, data)
result = fit.execute()

print(result.parameters)

>>> {'a': array(0.47572707), 'b': array(2.6199133)}

```


Installation
------------

```console
$ pip install slimfit
```
