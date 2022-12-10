from __future__ import annotations

import abc
import time
from dataclasses import asdict
from functools import reduce, cached_property
from operator import or_
from typing import Optional, Any

import numpy as np
from scipy.optimize import minimize
from sympy import Symbol
from tqdm.auto import trange

from slimfit import Model, NumExprBase
from slimfit.fitresult import FitResult
from slimfit.loss import Loss
from slimfit.objective import ScipyObjective, pack, unpack, ScipyEMObjective

# from slimfit.models import NumericalModel
from slimfit.operations import Mul
from slimfit.parameter import Parameters, Parameter
from slimfit.utils import get_bounds, overlapping_model_parameters

# TODO parameter which needs to be inferred / set somehow
STATE_AXIS = -2

# dataclass?
class Minimizer(metaclass=abc.ABCMeta):
    def __init__(
        self,
        numerical_model: Model,
        parameters: Parameters,
        loss: Loss,
        xdata: dict[str, np.array],
        ydata: dict[str, np.array],
    ):
        if not numerical_model.numerical:
            raise ValueError("The given model should be numerical")

        self.model = numerical_model
        self.loss = loss
        self.xdata = xdata
        self.ydata = ydata
        self.parameters = parameters

    # subset of parameters which are fixed
    @cached_property
    def fixed_parameters(self) -> Parameters:
        return Parameters([p for p in self.parameters if p.fixed])

    @cached_property
    def free_parameters(self) -> Parameters:
        return Parameters([p for p in self.parameters if not p.fixed])

    @property
    def guess(self) -> dict[str, np.ndarray]:
        # todo also on self.parameters.guess
        raise NotImplementedError("Use `free_parameters.guess")

        print("guess on minimizer")
        for p in self.parameters:
            print(p.name, p.fixed)

        return {p.name: np.asarray(p.guess) for p in self.parameters if not p.fixed}

    def get_bounds(self) -> list[tuple[float | None, float | None]] | None:
        bounds = [(p.lower_bound, p.upper_bound) for p in self.free_parameters]

        if all((None, None) == b for b in bounds):
            return None
        else:
            return bounds

    @abc.abstractmethod
    def execute(self, **minimizer_options) -> FitResult:
        ...


class ScipyMinimizer(Minimizer):
    def execute(self, **minimizer_options):
        param_shapes = {p.name: p.shape for p in self.free_parameters}

        objective = ScipyObjective(
            model=self.model,
            loss=self.loss,
            xdata=self.xdata | self.fixed_parameters.guess,
            ydata=self.ydata,
            shapes=param_shapes,
        )

        x = pack(self.free_parameters.guess.values())

        result = minimize(
            objective, x, bounds=self.get_bounds(), options=self.rename_options(minimizer_options)
        )

        gof_qualifiers = {
            "loss": result["fun"],
        }

        parameter_values = unpack(result.x, param_shapes)
        result_dict = dict(
            parameters=parameter_values,
            fixed_parameters=self.fixed_parameters,
            gof_qualifiers=gof_qualifiers,
            guess=self.free_parameters.guess,
            base_result=result,
        )

        # todo pass to superclass generalize fitresult function
        return FitResult(**result_dict)

    def rename_options(self, options: dict[str, Any]) -> dict[str, Any]:
        # todo parse options more generally
        rename = [("max_iter", "maxiter")]

        out = options.copy()
        out.pop("stop_loss", None)
        return out


# should take an optional CompositeNumExpr which returns the posterior
# todo refactor to EM Optimizer?
class LikelihoodOptimizer(Minimizer):
    """
    Assumed `loss` is `LogLoss`

    """

    def execute(self, max_iter=250, patience=5, stop_loss=1e-7, verbose=True) -> FitResult:
        # parameters which needs to be passed / inferred
        # Split top-level multiplications in the model as they can be optimized in log likelihood independently
        components: list[tuple[Symbol, NumExprBase]] = []  # todo tuple LHS as variable
        for lhs, rhs in self.model.items():
            if isinstance(rhs, Mul):
                components += [(lhs, elem) for elem in rhs.values()]
            else:
                components.append((lhs, rhs))

        # Find the sets of components which have common parameters and thus need to be optimized together
        sub_models = overlapping_model_parameters(components, self.free_parameters.symbols)

        pbar = trange(max_iter, disable=not verbose)
        t0 = time.time()

        parameters_current = self.free_parameters.guess  # initialize parameters
        prev_loss = 0.0
        no_progress = 0
        base_result = {}
        for i in pbar:
            y_model = self.model(**parameters_current, **self.xdata, **self.fixed_parameters.guess)
            loss = self.loss(self.ydata, y_model)

            # posterior dict has values with shapes equal to y_model
            # which is (dataopints, states, 1)
            posterior = {k: v / v.sum(axis=STATE_AXIS, keepdims=True) for k, v in y_model.items()}

            # dictionary of new parameter values for in this iteration
            parameters_step = {}
            common_kwargs = dict(
                loss=self.loss, xdata=self.xdata, ydata=self.ydata, posterior=posterior,
            )
            for sub_model in sub_models:
                # At the moment we assume all callables in the sub models to be MatrixCallables
                # determine the kind
                kinds = [c.kind for c in sub_model.values()]
                # Filter the general list of parameters to reduce it down to the parameters
                # this model accepts
                # the sub model also needs to the fixed parameters to correctly be able to
                # call the model; GMM also needs it for finding sigma
                sub_parameters = sub_model.filter_parameters(self.parameters)
                if all(k == "constant" for k in kinds):
                    # Constant optimizer doesnt use loss, xdata, ydata,
                    opt = ConstantOptimizer(sub_model, sub_parameters, **common_kwargs)
                    parameters = opt.step()
                elif all(k == "gmm" for k in kinds):
                    # Loss is not used for GMM optimizer step
                    opt = GMMOptimizer(sub_model, sub_parameters, **common_kwargs)
                    parameters = opt.step()
                else:

                    # Previous code:
                    # updated_parameters = [
                    #     Parameter(**(asdict(p) | {"guess": parameters_current.get(p.name) or self.fixed_parameters.guess[p.name]  }))
                    #     for p in sub_parameters
                    # ]

                    # New; needs improvement as it accesses `_names`
                    current_sub = {
                        k: v for k, v in parameters_current.items() if k in sub_parameters._names
                    }
                    updated_parameters = sub_parameters.update_guess(current_sub)

                    # todo loss is not used; should be EM loss while the main loop uses Log likelihood loss
                    opt = ScipyEMOptimizer(sub_model, updated_parameters, **common_kwargs)

                    scipy_result = opt.execute()
                    parameters = scipy_result.parameters
                    base_result["scipy"] = scipy_result

                # collect parameters of this sub_model into parmaeters dict
                parameters_step |= parameters

            # update for next iteration
            parameters_current = parameters_step

            # loss
            improvement = prev_loss - loss
            prev_loss = loss
            pbar.set_postfix({"improvement": improvement})
            if np.isnan(improvement):
                break
            elif improvement < stop_loss:
                no_progress += 1
            else:
                no_progress = 0

            if no_progress > patience:
                break

        tdelta = time.time() - t0
        gof_qualifiers = {
            "loss": loss,
            "log_likelihood": -loss,
            "n_iter": i + 1,
            "elapsed": tdelta,
            "iter/s": tdelta / (i + 1),
        }

        result = FitResult(
            parameters=parameters_current,
            fixed_parameters=self.fixed_parameters.guess,
            gof_qualifiers=gof_qualifiers,
            guess=self.free_parameters.guess,
            base_result=base_result,
        )

        return result


class EMOptimizer(Minimizer):
    def __init__(
        self,
        numerical_model: Model,
        parameters: list[Parameter],  # Parameters?
        loss: Loss,
        xdata: dict[str, np.array],
        ydata: dict[str, np.array],
        posterior: dict[str, np.array],
    ):
        self.posterior = posterior
        super().__init__(
            numerical_model=numerical_model,
            parameters=parameters,
            loss=loss,
            xdata=xdata,
            ydata=ydata,
        )

    @abc.abstractmethod
    def step(self) -> dict[str, float]:
        return {}

    def execute(self, max_iter=250, patience=5, stop_loss=1e-7, verbose=True) -> FitResult:
        raise NotImplementedError("Not yet")
        pbar = trange(max_iter, disable=not verbose)
        t0 = time.time()

        parameters_current = self.guess  # initialize parameters
        prev_loss = 0.0
        no_progress = 0
        # cache dict?
        for i in pbar:
            eval = self.model(**parameters_current)
            loss = self.loss(self.ydata, eval)

            parameters_step = self.step()

            # update parameters for next iteration
            parameters_current = parameters_step

            # Check improvement of loss
            improvement = prev_loss - loss
            prev_loss = loss
            pbar.set_postfix({"improvement": improvement})
            if np.isnan(improvement):
                break
            elif improvement < stop_loss:
                no_progress += 1
            else:
                no_progress = 0

            if no_progress > patience:
                break

        tdelta = time.time() - t0
        gof_qualifiers = {
            "loss": loss,
            "log_likelihood": -loss,
            "n_iter": i,
            "elapsed": tdelta,
            "iter/s": tdelta / i,
        }

        result = FitResult(
            parameters=parameters_current,
            fixed_parameters=self.model.fixed_parameters,
            gof_qualifiers=gof_qualifiers,
            guess=self.guess,
            # model=self.model,
            # data={**self.xdata, **self.ydata},
        )

        return result


class GMMOptimizer(EMOptimizer):
    """optimizes parameter values of GMM (sub) model
    doesnt use guess directly but instead finds next values for parmater from posterior probabilities

    rhs of model should all be GMM CompositeNumExprs

    """

    # TODO create `step` method which only does one step', execute should be full loop

    def step(self) -> dict[str, float]:
        parameters = {}  # output parameters dictionary

        # All symbols in all 'mu' expressions, then take intersection
        mu_symbols = set.union(*(rhs["mu"].symbols for rhs in self.model.values()))
        mu_symbols &= self.free_parameters.symbols

        # take only the mu symbos in the set of symbols designated as parameters
        mu_parameters = mu_symbols
        # mu_parameters = reduce(
        #     or_, [rhs["mu"].free_parameters.keys() for rhs in self.model.values()]
        # )
        for mu_symbol in mu_parameters:
            num, denom = 0.0, 0.0
            for lhs, gmm_rhs in self.model.items():
                # check if the current mu parameter in this GMM
                if mu_symbol in gmm_rhs["mu"].symbols:
                    col, state_index = gmm_rhs["mu"].index(mu_symbol)
                    T_i = np.take(self.posterior[str(lhs)], state_index, axis=STATE_AXIS)

                    # independent data should be given in the same shape as T_i
                    # which is typically (N, 1), to be sure shapes match we reshape independent data
                    # data = self.xdata[gmm_rhs['x'].name]
                    num += np.sum(T_i * self.xdata[gmm_rhs["x"].name].reshape(T_i.shape))
                    denom += np.sum(T_i)

            parameters[mu_symbol.name] = num / denom

        sigma_symbols = set.union(*(rhs["sigma"].symbols for rhs in self.model.values()))
        sigma_symbols &= self.free_parameters.symbols

        # sigma_parameters = reduce(
        #     or_, [rhs["sigma"].free_parameters.keys() for rhs in self.model.values()]
        # )
        for sigma_symbol in sigma_symbols:
            num, denom = 0.0, 0.0
            # LHS in numerical models are `str` (at the moment)
            for lhs, gmm_rhs in self.model.items():
                # check if the current sigma parameter in this GMM
                if sigma_symbol in gmm_rhs["sigma"].symbols:
                    col, state_index = gmm_rhs["sigma"].index(sigma_symbol)

                    T_i = np.take(self.posterior[str(lhs)], state_index, axis=STATE_AXIS)

                    # Indexing of the MatrixExpr returns elements of its expr
                    mu_name: str = gmm_rhs["mu"][col, state_index].name

                    # Take the corresponding value from the current parameters dict, if its not
                    # there, it must be in the fixed parameters of the model
                    try:
                        mu_value = parameters[mu_name]
                    except KeyError:
                        mu_value = self.fixed_parameters.guess[mu_name]

                    num += np.sum(
                        T_i * (self.xdata[gmm_rhs["x"].name].reshape(T_i.shape) - mu_value) ** 2
                    )

                    denom += np.sum(T_i)

            parameters[sigma_symbol.name] = np.sqrt(num / denom)

        return parameters


class ConstantOptimizer(EMOptimizer):

    """
    model is of form {Probability(px): Matrix[[<parameters>]] ...} where all matrix elements are just scalar parameters.
    """

    def step(self) -> dict[str, float]:
        parameters = {}
        # for p_name in self.model.free_parameters:
        for parameter in self.free_parameters:
            num, denom = 0.0, 0.0
            for lhs, rhs in self.model.items():
                # rhs is of type MatrixNumExpr
                if parameter.symbol in rhs.symbols:
                    # Shapes of RHS matrices is (N_states, 1), find index with .index(name)
                    state_index, _ = rhs.index(parameter.symbol)
                    T_i = np.take(self.posterior[str(lhs)], state_index, axis=STATE_AXIS)
                    num_i, denom_i = T_i.sum(), T_i.size

                    num += num_i
                    denom += denom_i
            parameters[parameter.name] = num / denom

        return parameters


class ScipyEMOptimizer(EMOptimizer):

    # TODO this is an abstract method
    def step(self):
        ...

    def execute(self, **minimizer_options):
        param_shapes = {p.name: p.shape for p in self.free_parameters}

        # x = self.model.parameters.pack(self.guess)
        # x = np.array([self.guess[p_name] for p_name in self.parameter_names])
        # options = {"method": "SLSQP"} | minimizer_options

        objective = ScipyEMObjective(
            model=self.model,
            loss=self.loss,  # not used currently
            xdata=self.xdata | self.fixed_parameters.guess,
            posterior=self.posterior,
            shapes=param_shapes,
        )

        x = pack(self.free_parameters.guess.values())
        options = minimizer_options
        # bounds = self.model.free_parameters.get_bounds()
        # todo what if users wants different bounds to pass to the minimizer?
        # perhaps that should also be passed, same as guess?
        # what about the pack / unpack?
        result = minimize(
            objective,
            x,
            # args=(self.model, self.loss, self.posterior),
            # args=(self.parameter_names, self.xdata, self.posterior, self.model, self.loss,),
            bounds=self.get_bounds(),
            **options
        )

        gof_qualifiers = {
            "loss": result["fun"],
        }

        parameter_values = unpack(result.x, param_shapes)

        result_dict = dict(
            parameters=parameter_values,
            fixed_parameters=self.fixed_parameters,
            gof_qualifiers=gof_qualifiers,
            guess=self.free_parameters.guess,
            base_result=result,
        )

        # todo pass to superclass generalize fitresult function
        # or functional
        return FitResult(**result_dict)

    # TODO duplicate code
    def to_fitresult(self, result) -> FitResult:

        parameters = self.model.parameters.unpack(result.x)

        gof_qualifiers = {
            "loss": result["fun"],
        }

        fit_result = FitResult(
            parameters=parameters,
            gof_qualifiers=gof_qualifiers,
            guess=self.guess,
            base_result=result,
        )

        return fit_result


MIN_PROB = 1e-9  # Minimal probability value (> 0.) to enter into np.log


def minfunc_expectation(
    x: np.ndarray,  # array of parameters
    model: Model,  # parameter names
    loss: Loss,  # currently not used
    posterior: dict,  # posterior probabilities
):
    params = model.parameters.unpack(x)
    probability = model(**params)

    # Todo do this in a `loss`
    expectation = {
        lhs: posterior[lhs] * np.log(np.clip(prob, a_min=MIN_PROB, a_max=1.0))
        for lhs, prob in probability.items()
    }
    # TODO: LOSS / WEIGHTS

    return -sum(r.sum() for r in expectation.values())


def minfunc(
    x: np.ndarray,  # array of parameters
    model: Model,
    loss: Loss,
    dependent_data: dict,  # corresponding measurements; target data
) -> float:

    parameter_values = model.parameters.unpack(x)
    predicted = model(**parameter_values)

    return loss(dependent_data, predicted)
