from __future__ import annotations

import abc
import time
from functools import reduce
from operator import or_
from typing import Optional, Any

import numpy as np
from scipy.optimize import minimize
from sympy import Symbol
from tqdm.auto import trange

from slimfit import Model, NumExprBase
from slimfit.fitresult import FitResult
from slimfit.loss import Loss
# from slimfit.models import NumericalModel
from slimfit.operations import Mul
from slimfit.parameter import Parameters
from slimfit.utils import get_bounds, overlapping_model_parameters

# TODO parameter which needs to be inferred / set somehow
STATE_AXIS = -2


class Minimizer(metaclass=abc.ABCMeta):
    def __init__(
        self, model: Model, loss: Loss, ydata: dict[str, np.array],
    ):
        if not model.numerical:
            raise ValueError("The given model should be numerical")

        self.model = model
        self.loss = loss
        self.ydata = ydata


    @abc.abstractmethod
    def execute(self, **minimizer_options) -> FitResult:
        ...


class ScipyMinimizer(Minimizer):
    def execute(self, **minimizer_options):
        x = self.model.parameters.pack()

        result = minimize(
            minfunc,
            x,
            args=(self.model, self.loss, self.ydata,),
            bounds=self.model.parameters.get_bounds(),
            **self.rename_options(minimizer_options)
        )

        return self.to_fitresult(result)

    def rename_options(self, options: dict[str, Any]) -> dict[str, Any]:
        # todo parse options more generally
        rename = [("max_iter", "maxiter")]

        # ("stop_loss", ""),
        out = options.copy()
        out.pop("stop_loss", None)
        return out

    def to_fitresult(self, result) -> FitResult:
        parameter_values = {
            name: arr.item() if arr.size == 1 else arr
            for name, arr in self.model.parameters.unpack(result.x).items()
        }

        gof_qualifiers = {
            "loss": result["fun"],
        }

        fit_result = FitResult(
            parameters=parameter_values,
            fixed_parameters=self.model.fixed_parameters,
            gof_qualifiers=gof_qualifiers,
            guess=self.model.parameters.guess,
            base_result=result,
        )

        return fit_result

# should take an optional CompositeNumExpr which returns the posterior
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
        sub_models = overlapping_model_parameters(components)

        pbar = trange(max_iter, disable=not verbose)
        t0 = time.time()

        parameters_current = self.model.parameters.guess  # initialize parameters
        prev_loss = 0.0
        no_progress = 0
        for i in pbar:
            result = self.model(**parameters_current)
            loss = self.loss(self.ydata, result)
            # posterior dict has values with shapes equal to eval
            # which is (dataopints, states, 1)
            posterior = {k: v / v.sum(axis=STATE_AXIS, keepdims=True) for k, v in result.items()}

            # At the moment we assume all callables in the sub models to be MatrixCallables
            # dictionary of new parameter values for in this iteration
            parameters_step = {}
            base_result = {}
            for sub_model in sub_models:
                # determine the kind
                kinds = [c.kind for c in sub_model.values()]
                if all(k == "constant" for k in kinds):
                    # Loss is not used for ConstantOptimizer step
                    opt = ConstantOptimizer(sub_model, self.loss, {}, posterior)
                    parameters = opt.step()
                elif all(k == "gmm" for k in kinds):
                    # Loss is not used for GMM optimizer step
                    opt = GMMOptimizer(sub_model, self.loss, {}, posterior)
                    parameters = opt.step()
                else:
                    raise NotImplementedError("Not yet")
                    guess = {k: parameters_current[k] for k in sub_model.free_parameters}
                    # todo loss is not used; should be EM loss while the main loop uses Log likelihood loss
                    opt = ScipyEMOptimizer(
                        sub_model, self.xdata, {}, posterior, loss=self.loss, guess=guess,
                    )
                    scipy_result = opt.execute()
                    parameters = scipy_result.parameters
                    base_result['scipy'] = scipy_result

                # collect parameters of this sub_model into parmaeters dict
                parameters_step |= parameters

            # update for next iteration
            parameters_current = parameters_step

            # loss
            improvement = prev_loss - loss
            prev_loss = loss
            pbar.set_postfix({"improvement": improvement})
            if improvement < stop_loss:
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
            fixed_parameters=self.model.fixed_parameters,
            gof_qualifiers=gof_qualifiers,
            guess=self.model.parameters.guess,
            base_result=base_result,
        )

        return result


class EMOptimizer(Minimizer):
    def __init__(
        self,
        model: Model,
        loss: Loss,
        ydata: dict[str, np.array],
        posterior: dict[str, np.array],
    ):
        self.posterior = posterior
        super().__init__(
            model=model, loss=loss, ydata=ydata,
        )

    @abc.abstractmethod
    def step(self) -> dict[str, float]:
        return {}

    def execute(self, max_iter=250, patience=5, stop_loss=1e-7, verbose=True) -> FitResult:

        pbar = trange(max_iter, disable=not verbose)
        t0 = time.time()

        parameters_current = self.model.parameters.guess  # initialize parameters
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
            if improvement < stop_loss:
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
            gof_qualifiers=gof_qualifiers,
            guess=self.model.parameters.guess,
            # model=self.model,
            # data={**self.xdata, **self.ydata},
        )

        return result


class GMMOptimizer(EMOptimizer):
    """optimizes parameter values of GMM (sub) model"""

    # TODO create `step` method which only does one step', execute should be full loop

    def step(self) -> dict[str, float]:
        parameters = {}  # output parameters dictionary

        mu_parameters = reduce(or_, [rhs['mu'].free_parameters.keys() for rhs in self.model.values()])
        for p_name in mu_parameters:
            num, denom = 0.0, 0.0
            for lhs, gmm_rhs in self.model.items():
                # check if the current mu parameter in this GMM
                if p_name in gmm_rhs['mu'].free_parameters:
                    col, state_index = gmm_rhs['mu'].index(p_name)
                    T_i = np.take(self.posterior[str(lhs)], state_index, axis=STATE_AXIS)

                    # independent data should be given in the same shape as T_i
                    # which is typically (N, 1), to be sure shapes match we reshape independent data
                    num += np.sum(T_i * self.model.data[gmm_rhs['x'].name].reshape(T_i.shape))
                    denom += np.sum(T_i)

            parameters[p_name] = num / denom

        sigma_parameters = reduce(
            or_, [rhs['sigma'].free_parameters.keys() for rhs in self.model.values()]
        )
        for p_name in sigma_parameters:
            num, denom = 0.0, 0.0
            # LHS in numerical models are `str` (at the moment)
            for lhs, gmm_rhs in self.model.items():
                # check if the current sigma parameter in this GMM
                if p_name in gmm_rhs['sigma'].free_parameters:
                    col, state_index = gmm_rhs['sigma'].index(p_name)

                    T_i = np.take(self.posterior[str(lhs)], state_index, axis=STATE_AXIS)

                    # Indexing of the MatrixExpr returns elements of its expr
                    mu_name: str = gmm_rhs['mu'][col, state_index].name

                    # Take the corresponding value from the current parameters dict, if its not
                    # there, it must be in the fixed parameters of the model
                    try:
                        mu_value = parameters[mu_name]
                    except KeyError:
                        mu_value = self.model.fixed_parameters[mu_name]

                    num += np.sum(
                        T_i * (self.model.data[gmm_rhs['x'].name].reshape(T_i.shape) - mu_value) ** 2
                    )


                    denom += np.sum(T_i)

            parameters[p_name] = np.sqrt(num / denom)

        return parameters


class ConstantOptimizer(EMOptimizer):

    """
    model is of form {Probability(px): Matrix[[<parameters>]] ...} where all matrix elements are just scalar parameters.
    """

    def step(self) -> dict[str, float]:
        parameters = {}
        for p_name in self.model.free_parameters:
            num, denom = 0.0, 0.0
            for lhs, rhs in self.model.items():
                # rhs is of type MatrixNumExpr
                if p_name in rhs.free_parameters:
                    # Shapes of RHS matrices is (N_states, 1), find index with .index(name)
                    state_index, _ = rhs.index(p_name)
                    T_i = np.take(self.posterior[str(lhs)], state_index, axis=STATE_AXIS)
                    num_i, denom_i = T_i.sum(), T_i.size

                    num += num_i
                    denom += denom_i
            parameters[p_name] = num / denom

        return parameters


class ScipyEMOptimizer(EMOptimizer):

    # TODO this is an abstract method
    def step(self):
        ...

    def execute(self, **minimizer_options):
        x = np.array([self.guess[p_name] for p_name in self.parameter_names])

        # options = {"method": "SLSQP"} | minimizer_options
        options = minimizer_options

        result = minimize(
            minfunc_expectation,
            x,
            args=(self.parameter_names, self.xdata, self.posterior, self.model, self.loss,),
            bounds=get_bounds(self.model.free_parameters.values()),
            **options
        )

        return self.to_fitresult(result)

    # TODO duplicate code
    def to_fitresult(self, result) -> FitResult:
        parameters = {k: xi for k, xi in zip(self.parameter_names, result.x)}

        gof_qualifiers = {
            "loss": result["fun"],
        }

        fit_result = FitResult(
            parameters=parameters,
            gof_qualifiers=gof_qualifiers,
            guess=self.guess,
            data={**self.xdata, **self.ydata},
            _result=result,
        )

        return fit_result


MIN_PROB = 1e-9  # Minimal probability value (> 0.) to enter into np.log


def minfunc_expectation(
    x: np.ndarray,  # array of parameters
    x_names,  # parameter names
    independent_data: dict,  # measurement points
    posterior: dict,  # posterior probabilities
    model: Model,
    loss: Loss,
):
    params = {name: value for name, value in zip(x_names, x)}
    probability = model(**independent_data, **params)

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
