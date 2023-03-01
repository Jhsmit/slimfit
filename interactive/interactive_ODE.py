import param
import panel as pn
import numpy as np
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from sympy import exp, Matrix

from slimfit import Model, Variable, Probability, Parameter
from slimfit.markov import generate_transition_matrix, extract_states

pn.extension(sizing_mode="stretch_width")


class InteractiveODE(param.Parameterized):
    states = param.List(doc="list of state names.")

    colors = param.List(doc="list of state colors.")

    log_params = param.List(doc="list of params to transform to log space.")

    save_params = param.Action(lambda self: self._action_save_params())

    def __init__(self, model: Model, data: dict, **params):
        super().__init__(**params)
        self.model = model
        self.data = data

        self.sliders = {}
        # dict of params without bounds or set to `fixed`
        self.fixed_params = {}
        for par in self.model.parameters.values():
            if par.fixed:
                self.fixed_params[par.name] = par.value
            elif par.vmax is not None and par.vmin is not None:
                if par.name in self.log_params:
                    value, vmin, vmax = np.log10([par.value, par.vmin, par.vmax])
                    label = f"{par.name} (log10)"
                else:
                    value, vmin, vmax = par.value, par.vmin, par.vmax
                    label = par.name

                slider = pn.widgets.FloatSlider(value=value, start=vmin, end=vmax, name=label)

                self.sliders[par.name] = slider
                slider.param.watch(self._slider_changed, "value")
            else:
                self.fixed_params[par.name] = par.value

        self.figure = figure(width=750, height=400)
        self.cds = ColumnDataSource(self.get_dict())
        self.init_graph()

    def init_graph(self) -> None:
        for color, state in zip(self.colors, self.states):
            self.figure.line(x="t", y=state, source=self.cds, color=color, line_width=4)

    def get_dict(self) -> dict:
        kwargs = {}
        for name, slider in self.sliders.items():
            if name in self.log_params:
                kwargs[name] = 10**slider.value
            else:
                kwargs[name] = slider.value
        res = self.model(**self.data, **kwargs)

        data_dict = {st: f_j for st, f_j in zip(self.states, res["p"].squeeze().T)}
        data_dict |= self.data

        return data_dict

    def _slider_changed(self, event):
        self.cds.data = self.get_dict()


connectivity = ["A <-> B -> C"]
m = generate_transition_matrix(connectivity)
states = extract_states(connectivity)

# Set rate bounds
Parameter("k_A_B", vmin=1e-2, vmax=1e3)
Parameter("k_B_A", vmin=1e-2, vmax=1e3)
Parameter("k_B_C", vmin=1e-2, vmax=1e3)

y0 = [
    Parameter("y0_A", value=1.0, vmin=0.0, vmax=1.0),
    Parameter("y0_B", value=0.0, vmin=0.0, vmax=1.0),
    Parameter("y0_C", value=0.0, vmin=0.0, vmax=1.0),
]
y0 = Matrix([y0]).T

# exponentiate the transition rate matrix
xt = exp(m * Variable("t"))
model = Model({Probability("p"): xt @ y0})

tdata = np.linspace(0, 11, num=250)
data = {"t": tdata}

ode = InteractiveODE(
    model,
    data,
    states=states,
    colors=["red", "cyan", "green"],
    log_params=["k_A_B", "k_B_A", "k_B_C"],
)

accent_color = "#15590b"
app = pn.template.FastListTemplate(
    title="Interactive ODEs",
    header_background=accent_color,
    sidebar=list(ode.sliders.values()),
    main=[pn.pane.Bokeh(ode.figure)],
)

if __name__.startswith("__main__"):
    pn.serve(app)
elif __name__.startswith("bokeh"):
    app.servable()
