import abc
import time
from functools import reduce
from operator import or_
from typing import Optional, Any

import numpy as np
from scipy.optimize import minimize
from tqdm.auto import trange

from slimfit import Model, NumExprBase
from slimfit.fitresult import FitResult
from slimfit.loss import Loss
from slimfit.operations import Mul
from slimfit.symbols import FitSymbol
from slimfit.utils import get_bounds, overlapping_model_parameters


# TODO parameter which needs to be inferred / set somehow
STATE_AXIS = -2


class Minimizer(metaclass=abc.ABCMeta):
    def __init__(
        self,
        model: Model,
        independent_data: dict[str, np.array],
        dependent_data: dict[str, np.array],
        loss: Loss,
        guess: Optional[dict[str, float]] = None,
    ):
        self.loss = loss
        self.model = model
        self.independent_data = independent_data
        self.dependent_data = dependent_data
        guess = guess or {}
        self.guess = self.model.guess | guess

    @abc.abstractmethod
    def execute(self, **minimizer_options) -> FitResult:
        ...

    @property
    def parameter_names(self) -> list[str]:
        """List of parameter names in the model"""
        return list(self.model.free_parameters.keys())


class ScipyMinimizer(Minimizer):
    def execute(self, **minimizer_options):
        x = np.array([self.guess[p_name] for p_name in self.parameter_names])

        result = minimize(
            minfunc,
            x,
            args=(
                self.parameter_names,
                self.independent_data,
                self.dependent_data,
                self.model,
                self.loss,
            ),
            bounds=get_bounds(self.model.free_parameters.values()),
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
        parameters = {k: xi for k, xi in zip(self.parameter_names, result.x)}

        gof_qualifiers = {
            "loss": result["fun"],
        }

        fit_result = FitResult(
            parameters=parameters,
            gof_qualifiers=gof_qualifiers,
            guess=self.guess,
            data={**self.independent_data, **self.dependent_data},
            _result=result,
        )

        return fit_result


class LikelihoodOptimizer(Minimizer):
    """
    Assumed `loss` is `LogLoss`

    """

    def execute(self, max_iter=250, patience=5, stop_loss=1e-7, verbose=True) -> FitResult:
        # parameters which needs to be passed / inferred

        # Split top-level multiplications in the model as they can be optimized in log likelihood independently
        components: list[tuple[FitSymbol, NumExprBase]] = []  # todo tuple LHS as variable
        for lhs, rhs in self.model.items():
            if isinstance(rhs, Mul):
                components += [(lhs, elem) for elem in rhs.elements]
            else:
                components.append((lhs, rhs))

        # Find the sets of components which have common parameters and thus need to be optimized together
        sub_models = overlapping_model_parameters(components)
        # loop here

        pbar = trange(max_iter, disable=not verbose)
        t0 = time.time()

        parameters_current = self.guess  # initialize parameters
        prev_loss = 0.0
        no_progress = 0
        for i in pbar:
            eval = self.model(**self.independent_data, **parameters_current)
            loss = self.loss(self.dependent_data, eval)
            # posterior dict has values with shapes equal to eval
            posterior = {k: v / v.sum(axis=STATE_AXIS, keepdims=True) for k, v in eval.items()}

            # At the moment we assume all callables in the sub models to be MatrixCallables
            # dictionary of new parameter values for in this iteration
            parameters_step = {}
            for sub_model in sub_models:
                # determine the kind
                kinds = [c.kind for c in sub_model.values()]
                if all([k == "constant" for k in kinds]):
                    opt = ConstantOptimizer(
                        sub_model, self.independent_data, {}, posterior, loss=self.loss
                    )
                    parameters = opt.step()
                elif all([k == "gmm" for k in kinds]):
                    opt = GMMOptimizer(
                        sub_model, self.independent_data, {}, posterior, loss=self.loss
                    )
                    parameters = opt.step()
                else:
                    guess = {k: parameters_current[k] for k in sub_model.free_parameters}
                    # todo loss is not used; should be EM loss while the main loop uses Log likelihood loss
                    opt = ScipyEMOptimizer(
                        sub_model,
                        self.independent_data,
                        {},
                        posterior,
                        loss=self.loss,
                        guess=guess,
                    )
                    result = opt.execute()
                    parameters = result.parameters

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
            "n_iter": i,
            "elapsed": tdelta,
            "iter/s": tdelta / i,
        }

        result = FitResult(
            parameters=parameters_current,
            gof_qualifiers=gof_qualifiers,
            guess=self.guess,
            model=self.model,
            data={**self.independent_data, **self.dependent_data},
        )

        return result


class EMOptimizer(Minimizer):
    def __init__(
        self,
        model: Model,
        independent_data: dict[str, np.array],
        dependent_data: dict[str, np.array],
        posterior: dict[str, np.array],
        loss: Loss,
        guess: Optional[dict[str, float]] = None,
    ):
        self.posterior = posterior
        super().__init__(
            model=model,
            independent_data=independent_data,
            dependent_data=dependent_data,
            loss=loss,
            guess=guess,
        )

    @abc.abstractmethod
    def step(self) -> dict[str, float]:
        return {}

    def execute(self, max_iter=250, patience=5, stop_loss=1e-7, verbose=True) -> FitResult:

        pbar = trange(max_iter, disable=not verbose)
        t0 = time.time()

        parameters_current = self.guess  # initialize parameters
        prev_loss = 0.0
        no_progress = 0
        # cache dict?
        for i in pbar:
            eval = self.model(**self.independent_data, **parameters_current)
            loss = self.loss(self.dependent_data, eval)

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
            guess=self.guess,
            model=self.model,
            data={**self.independent_data, **self.dependent_data},
        )

        return result


class GMMOptimizer(EMOptimizer):
    """optimizes parameter values of GMM (sub) model"""

    # TODO create `step` method which only does one step', execute should be full loop

    def step(self) -> dict[str, float]:
        parameters = {}  # output parameters dictionary

        mu_parameters = reduce(or_, [rhs.mu.free_parameters.keys() for rhs in self.model.values()])
        for p_name in mu_parameters:
            num, denom = 0.0, 0.0
            for lhs, gmm_rhs in self.model.items():
                # check if the curret mu parameter in this GMM
                if p_name in gmm_rhs.mu.parameters:
                    state_index, col = gmm_rhs.mu.index(p_name)
                    T_i = np.take(self.posterior[lhs.name], state_index, axis=STATE_AXIS)

                    # independent data should be given in the same shape as T_i
                    # which is typically (N, 1), to be sure shapes match we reshape independent data
                    num += np.sum(T_i * self.independent_data[gmm_rhs.x.name].reshape(T_i.shape))
                    denom += np.sum(T_i)

            parameters[p_name] = num / denom

        sigma_parameters = reduce(
            or_, [rhs.sigma.free_parameters.keys() for rhs in self.model.values()]
        )
        for p_name in sigma_parameters:
            num, denom = 0.0, 0.0
            for lhs, gmm_rhs in self.model.items():
                # check if the current sigma parameter in this GMM
                if p_name in gmm_rhs.sigma.parameters:
                    state_index, col = gmm_rhs.sigma.index(p_name)
                    T_i = np.take(self.posterior[lhs.name], state_index, axis=STATE_AXIS)

                    mu_parameter = gmm_rhs.mu[state_index, col]
                    mu_value = (
                        mu_parameter.value if mu_parameter.fixed else parameters[mu_parameter.name]
                    )

                    num += np.sum(
                        T_i
                        * (self.independent_data[gmm_rhs.x.name].reshape(T_i.shape) - mu_value) ** 2
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
        for p_name in self.model.parameters:
            nom, denom = 0.0, 0.0
            for lhs, rhs in self.model.items():
                if p_name in rhs.parameters:
                    # Shapes of RHS matrices is (N_states, 1), find index with .index(name)
                    state_index, _ = rhs.index(p_name)
                    T_i = np.take(self.posterior[lhs.name], state_index, axis=STATE_AXIS)
                    nom_i, denom_i = T_i.sum(), T_i.size

                    nom += nom_i
                    denom += denom_i
            parameters[p_name] = nom / denom

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
            args=(
                self.parameter_names,
                self.independent_data,
                self.posterior,
                self.model,
                self.loss,
            ),
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
            data={**self.independent_data, **self.dependent_data},
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
    x_names,  # parameter names
    independent_data: dict,  # measurement points
    dependent_data: dict,  # corresponding measurements; target data
    model: Model,
    loss: Loss,
) -> float:
    params = {name: value for name, value in zip(x_names, x)}
    predicted = model(**independent_data, **params)

    return loss(dependent_data, predicted)
