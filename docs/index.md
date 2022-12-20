# SlimFit Documentation

## Quick Start 


``` python
import tensorflow as tf
```


``` python

from sympy import symbols
from slimfit import Model, Fit, Parameter
import numpy as np

# Generate some data

xdata = np.linspace(0, 11, 25)
ydata = 0.5*xdata + 2.5
ydata += np.random.normal(0, scale= ydata / 10.0 + 0.2)
data = {'x': xdata, 'y': ydata}

# Define model and parameters
y, a, x, b = symbols('y a x b')
model = Model({y: a*x + b})
parameters = [
    Parameter(a, guess=2.5),
    Parameter(b, guess=1, lower_bound=0.)
]

# Fit the model
fit = Fit(model, parameters, data)
result = fit.execute()

print(result.parameters)

```

Lorem ipsum