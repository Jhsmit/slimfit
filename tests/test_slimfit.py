from pathlib import Path

import pytest

# from dont_fret.em_fit.datagen import generate_dataset
from sympy import HadamardProduct, Matrix, exp, Symbol

# import numpy as np

from slimfit.fit import Fit
from slimfit.functions import gaussian, gaussian_sympy, gaussian_numpy
from slimfit.loss import LogLoss

# from slimfit.markov import generate_transition_matrix, extract_states
# from slimfit.minimizer import LikelihoodOptimizer
from slimfit.operations import Mul, MatMul
from slimfit.models import Model
from slimfit.numerical import MatrixNumExpr, NumExpr, GMM, to_numerical, LambdaNumExpr, MarkovIVP
from slimfit.symbols import (
    symbol_matrix,
    clear_symbols,
    get_symbols,
)
from slimfit.loss import LogSumLoss
from slimfit.minimizer import LikelihoodOptimizer
from slimfit.markov import generate_transition_matrix, extract_states
from slimfit.parameter import Parameters, Parameter
import numpy as np

root_dir = Path(__file__).parent.parent


class TestEMBase(object):
    def test_symbol_matrix(self):
        clear_symbols()
        m = symbol_matrix("A", shape=(3, 3))

        assert m.shape == (3, 3)
        elem = m[0, 0]
        assert elem.name == "A_0_0"

        m = symbol_matrix("A", shape=(1, 3), suffix=["a", "b", "c"],)

        elem = m[0, 0]
        assert elem.name == "A_a"

        elem = m[0, 1]
        assert elem.name == "A_b"

        parameters = {
            "A_a": Parameter(m[0, 0], guess=2.0),
            "A_b": Parameter(m[0, 1]),
            "A_c": Parameter(m[0, 2]),
        }

        m_num = to_numerical(m)
        values = {"A_a": 1, "A_b": 2, "A_c": 3.5}

        result = m_num(**values)
        assert result.shape == (1, 3)
        assert np.allclose(result, np.array([1.0, 2.0, 3.5]).reshape(1, 3))

    # @pytest.mark.skip("Old test")
    def test_model(self):
        clear_symbols()

        # Create GMM from sympy operations and Parameters
        suffix = ["1", "2", "3"]
        mu = symbol_matrix(name="mu", shape=(3, 1), suffix=suffix)
        sigma = symbol_matrix(name="sigma", shape=(3, 1), suffix=suffix)

        # elementwise gaussian
        g = gaussian(Symbol("x"), mu, sigma)

        c = symbol_matrix(name="c", shape=(3, 1), suffix=suffix)
        model_dict = {Symbol("p"): HadamardProduct(c, g)}

        model = Model(model_dict)
        num_model = model.to_numerical()
        rhs = next(iter(num_model.values()))
        assert isinstance(rhs, Mul)

        symbols = get_symbols(mu, sigma, c)
        parameters = Parameters.from_symbols(symbols.values())
        print(parameters.guess)

        # Test calling the model
        kwargs = {"x": np.linspace(0, 1, num=100), **parameters.guess}
        res = num_model(**kwargs)

        assert res["p"].shape == (100, 3, 1)

        val = sum(np.trapz(f, kwargs["x"]) for f in res["p"].squeeze().T)
        assert 1 == pytest.approx(val, 1)


class TestNumExpr(object):
    def test_num_expr(self):
        clear_symbols()
        np.random.seed(43)

        x = np.arange(100).reshape(-1, 1)
        data = {"x": x}
        parameters = {
            "a": Parameter(Symbol("a"), guess=np.array([1, 2, 3]).reshape(1, -1)),
            "b": Parameter(Symbol("b"), guess=5.0),
        }

        expr = Symbol("a") * Symbol("x") + Symbol("b")
        num_expr = NumExpr(expr)

        # todo test shapes
        # assert num_expr.shape == (100, 3)

        a = np.random.rand(1, 3)
        b = 5.0
        result = num_expr(a=a, b=b, **data)
        assert np.allclose(result, a * x + b)

    def test_matrix_num_expr(self):
        clear_symbols()

        m = Matrix(
            [
                [
                    Symbol("a") * Symbol("x") + Symbol("b1"),
                    Symbol("a") * Symbol("x") + Symbol("b2"),
                    Symbol("a") * Symbol("x") + Symbol("b3"),
                ]
            ]
        )

        data = {"x": np.arange(100).reshape(-1, 1)}
        symbols = get_symbols(m)

        parameters = {
            "a": Parameter(symbols["a"], guess=np.random.rand(1, 3)),
            "b1": Parameter(symbols["b1"], guess=1.0),
            "b2": Parameter(symbols["b1"], guess=2.0),
            "b3": Parameter(symbols["b1"], guess=3.0),
        }

        m_expr = MatrixNumExpr(m)
        # todo shapes
        # assert m_expr.shape

        p_values = {
            "a": np.array([3, 2, 1]).reshape(1, -1),
            "b1": 2.0,
            "b2": 3.0,
            "b3": 4.0,
        }

        result = m_expr(**p_values, **data)

        # v assert result.shape == m_expr.shape

        check = data["x"] * p_values["a"] + p_values["b1"]
        assert np.allclose(check, result[..., 0, 0])

        check = data["x"] * p_values["a"] + p_values["b2"]
        assert np.allclose(check, result[..., 0, 1])

        check = data["x"] * p_values["a"] + p_values["b3"]
        assert np.allclose(check, result[..., 0, 2])

        # test symbol matrix factory function
        shape = (1, 3)
        states = ["A", "B", "C"]
        c = symbol_matrix(name="c", shape=shape, suffix=states)
        assert c[0, 0] == Symbol("c_A")

        num_c = to_numerical(c)

        assert num_c.kind == "constant"
        assert num_c.name == "c"

    def test_lambda_numexpr(self):
        clear_symbols()
        np.random.seed(43)

        def func(x, a):
            return x ** 2 + a

        data = {"x": np.arange(100)}

        ld = LambdaNumExpr(func, [Symbol("a"), Symbol("x")],)

        # todo shape testing
        # assert ld.shape == (100,)

        result = ld(a=2.0, **data)
        assert np.allclose(result, data["x"] ** 2 + 2.0)

    def test_gmm(self):
        states = ["A", "B", "C"]
        mu = symbol_matrix("mu", suffix=states, shape=(1, 3))
        sigma = symbol_matrix("sigma", suffix=states, shape=(1, 3))
        gmm = GMM(Symbol("x"), mu, sigma)
        parameters = Parameters.from_symbols(gmm.symbols, "mu_A mu_B mu_C sigma_A sigma_B sigma_C")
        data = {"x": np.linspace(-0.2, 1.2, num=25).reshape(-1, 1)}

        gt = {
            "mu_A": 0.23,
            "mu_B": 0.55,
            "mu_C": 0.92,
            "sigma_A": 0.1,
            "sigma_B": 0.1,
            "sigma_C": 0.1,
            "c_A": 0.22,
            "c_B": 0.53,
            "c_C": 0.25,
        }

        num_gmm = gmm.to_numerical()
        assert gmm.kind == "gmm"
        # todo shapes
        # assert num_gmm.shape == (25, 3, 1)
        assert isinstance(num_gmm["mu"], MatrixNumExpr)

        result = num_gmm(**gt, **data)
        assert result.shape == (25, 3, 1)


class TestEMFit(object):
    def test_linear_lstsq(self):
        clear_symbols()
        np.random.seed(43)

        gt = {"a": 0.5, "b": 2.5}

        xdata = np.linspace(0, 11, num=100)
        ydata = gt["a"] * xdata + gt["b"]

        noise = np.random.normal(0, scale=ydata / 10.0 + 0.5)
        ydata += noise

        data = {"x": xdata, "y": ydata}

        model = Model({Symbol("y"): Symbol("a") * Symbol("x") + Symbol("b")})
        parameters = Parameters.from_symbols(model.symbols, "a b")
        fit = Fit(model, parameters, data)
        res = fit.execute()

        assert res.parameters["a"] == pytest.approx(gt["a"], abs=0.2)
        assert res.parameters["b"] == pytest.approx(gt["b"], abs=0.1)

    def test_likelihood_gaussian(self):
        clear_symbols()
        np.random.seed(43)

        gt = {"mu": 2.4, "sigma": 0.7}

        data = {"x": np.random.normal(gt["mu"], scale=gt["sigma"], size=100)}
        model = Model({Symbol("p"): gaussian_sympy(Symbol("x"), Symbol("mu"), Symbol("sigma"))})

        parameters = Parameters.from_symbols(model.symbols, "mu sigma")

        fit = Fit(model, parameters, data, loss=LogLoss())
        res = fit.execute()

        assert res.parameters["mu"] == pytest.approx(gt["mu"], abs=0.05)
        assert res.parameters["sigma"] == pytest.approx(gt["sigma"], abs=0.05)

    def test_linear_matrix(self):
        clear_symbols()
        np.random.seed(43)

        mu_vals = [1.1, 3.5, 7.2]
        sigma_vals = [0.25, 0.1, 0.72]
        wavenumber = np.linspace(0, 11, num=100)  # wavenumber x-axis
        basis = np.stack(
            [gaussian_numpy(wavenumber, mu_i, sig_i) for mu_i, sig_i in zip(mu_vals, sigma_vals)]
        ).T

        x_vals = np.array([0.3, 0.5, 0.2]).reshape(3, 1)  # unknowns
        spectrum = basis @ np.array(x_vals).reshape(3, 1)  # measured

        x = symbol_matrix(name="X", shape=(3, 1))
        symbols = get_symbols(x)
        parameters = Parameters.from_symbols(symbols.values())

        model = Model({Symbol("b"): MatMul(basis, x)})
        fit = Fit(model, parameters, data={"b": spectrum})
        result = fit.execute()

        for i, j in np.ndindex(x_vals.shape):
            assert x_vals[i, j] == pytest.approx(result.parameters[f"X_{i}_{j}"], rel=1e-3)

    def test_exponential_matrix(self):
        clear_symbols()
        np.random.seed(43)

        gt_values = {
            "k_A_B": 1e0,
            "k_B_A": 5e-2,
            "k_B_C": 5e-1,
            "y0_A": 1.0,
            "y0_B": 0.0,
            "y0_C": 0.0,
        }

        connectivity = ["A <-> B -> C"]
        m = generate_transition_matrix(connectivity)
        states = extract_states(connectivity)

        xt = exp(m * Symbol("t"))
        y0 = symbol_matrix(name="y0", shape=(3, 1), suffix=states)
        model = Model({Symbol("y"): xt @ y0})

        rate_params = Parameters.from_symbols(get_symbols(m).values())
        y0_params = Parameters.from_symbols(get_symbols(y0).values())
        parameters = rate_params + y0_params

        num = 50
        xdata = {"t": np.linspace(0, 11, num=num)}
        num_model = model.to_numerical()
        populations = num_model(**gt_values, **xdata)["y"]
        ydata = {
            "y": populations + np.random.normal(0, 0.05, size=num * 3).reshape(populations.shape)
        }

        fit = Fit(model, parameters, data={**xdata, **ydata})
        result = fit.execute()

        expected = {
            "k_A_B": 1.0926495267297978,
            "k_B_A": 0.02553115392319696,
            "k_B_C": 0.48848195581215753,
            "y0_A": 1.0144580699136068,
            "y0_B": -0.011557732388925912,
            "y0_C": -0.006383620511149652,
        }

        for k in expected:
            assert result.parameters[k] == pytest.approx(expected[k], rel=0.1)

    def test_markov_ivp(self):
        clear_symbols()
        np.random.seed(43)

        gt_values = {
            "k_A_B": 1e0,
            "k_B_A": 5e-2,
            "k_B_C": 5e-1,
            "y0_A": 1.0,
            "y0_B": 0.0,
            "y0_C": 0.0,
        }

        connectivity = ["A <-> B -> C"]
        m = generate_transition_matrix(connectivity)
        states = extract_states(connectivity)

        y0 = symbol_matrix(name="y0", shape=(3, 1), suffix=states)
        model = Model({Symbol("y"): MarkovIVP(Symbol("t"), m, y0)})

        rate_params = Parameters.from_symbols(get_symbols(m).values())
        y0_params = Parameters.from_symbols(get_symbols(y0).values())
        parameters = rate_params + y0_params

        num = 50
        xdata = {"t": np.linspace(0, 11, num=num)}

        num_model = model.to_numerical()
        populations = num_model(**gt_values, **xdata)["y"]

        ydata = {
            "y": populations + np.random.normal(0, 0.05, size=num * 3).reshape(populations.shape)
        }
        ydata["y"].shape  # shape of the data is (50, 3, 1)

        fit = Fit(model, parameters, data={**xdata, **ydata})
        result = fit.execute()

        expected = {
            "k_A_B": 1.0928473669526968,
            "k_B_A": 0.025697062159894243,
            "k_B_C": 0.4884426216496761,
            "y0_A": 1.0144463997001603,
            "y0_B": -0.011475371097876597,
            "y0_C": -0.006402646431591074,
        }

        for k in expected:
            assert result.parameters[k] == pytest.approx(expected[k], rel=0.1)

    def test_gmm(self):
        clear_symbols()
        np.random.seed(43)

        gt = {
            "mu_A": 0.23,
            "mu_B": 0.55,
            "mu_C": 0.92,
            "sigma_A": 0.1,
            "sigma_B": 0.1,
            "sigma_C": 0.1,
            "c_A": 0.22,
            "c_B": 0.53,
            "c_C": 0.25,
        }

        np.random.seed(43)
        N = 1000
        states = ["A", "B", "C"]
        xdata = np.concatenate(
            [
                np.random.normal(
                    loc=gt[f"mu_{s}"], scale=gt[f"sigma_{s}"], size=int(N * gt[f"c_{s}"])
                )
                for s in states
            ]
        )

        np.random.shuffle(xdata)
        data = {"x": xdata.reshape(-1, 1)}

        guess = {
            "mu_A": 0.2,
            "mu_B": 0.4,
            "mu_C": 0.7,
            "sigma_A": 0.1,
            "sigma_B": 0.1,
            "sigma_C": 0.1,
            "c_A": 0.33,
            "c_B": 0.33,
            "c_C": 0.33,
        }

        g_shape = (1, 3)
        c_shape = (3, 1)
        mu = symbol_matrix(name="mu", shape=g_shape, suffix=states)
        sigma = symbol_matrix(name="sigma", shape=g_shape, suffix=states)
        c = symbol_matrix(name="c", shape=c_shape, suffix=states)

        model = Model({Symbol("p"): Mul(c, GMM(Symbol("x"), mu, sigma))})

        symbols = get_symbols(mu, sigma, c)
        parameters = Parameters.from_symbols(symbols.values(), guess)

        fit = Fit(model, parameters, data, loss=LogSumLoss(sum_axis=1))
        result = fit.execute(minimizer=LikelihoodOptimizer, verbose=False)

        expected = {
            "c_A": 0.21807944048774222,
            "c_B": 0.5351105112590985,
            "c_C": 0.24681004825315928,
            "mu_A": 0.23155221598099554,
            "mu_B": 0.5508567564172897,
            "mu_C": 0.9204744537231175,
            "sigma_A": 0.09704934271877938,
            "sigma_B": 0.09910459765563108,
            "sigma_C": 0.09877267156818363,
        }

        for k in expected:
            assert result.parameters[k] == pytest.approx(expected[k], rel=0.1)

        # Repeat with fixed parameters

        # fix mu A
        parameters.set("mu_A", fixed=True)
        fit = Fit(model, parameters, data, loss=LogSumLoss(sum_axis=1))
        result = fit.execute(minimizer=LikelihoodOptimizer, verbose=False)

        expected = {
            "c_A": 0.18205289103947245,
            "c_B": 0.5822954009717819,
            "c_C": 0.23565170798874566,
            "mu_B": 0.5427685760300586,
            "mu_C": 0.9273517146087585,
            "sigma_A": 0.08436777980742473,
            "sigma_B": 0.11294214298924928,
            "sigma_C": 0.09474026756245091,
        }
        for k in expected:
            assert result.parameters[k] == pytest.approx(expected[k], rel=0.1)

        for fixed_param in ["mu_A"]:
            assert result.fixed_parameters[fixed_param] == guess[fixed_param]

        # fix sigma B
        parameters.set("sigma_B", fixed=True)
        fit = Fit(model, parameters, data, loss=LogSumLoss(sum_axis=1))
        result = fit.execute(minimizer=LikelihoodOptimizer)

        expected = {
            "c_A": 0.1940912954966998,
            "c_B": 0.5552712770246595,
            "c_C": 0.25063742747864065,
            "mu_B": 0.5418477340418327,
            "mu_C": 0.9177305634612395,
            "sigma_A": 0.08718283966513221,
            "sigma_C": 0.10072652395544755,
        }
        for k in expected:
            assert result.parameters[k] == pytest.approx(expected[k], rel=0.1)

        for fixed_param in ["mu_A", "sigma_B"]:
            assert result.fixed_parameters[fixed_param] == guess[fixed_param]

    def test_global_gmm(self):
        """Test fitting of multiple GMM datasets with overlapping populations"""
        clear_symbols()
        np.random.seed(43)

        states = ["ABC", "BCD"]

        gt = {
            "mu_A": 0.23,
            "mu_B": 0.55,
            "mu_C": 0.92,
            "mu_D": 0.32,
            "sigma_A": 0.1,
            "sigma_B": 0.1,
            "sigma_C": 0.1,
            "sigma_D": 0.2,
            "c_A": 0.22,
            "c_B": 0.53,
            "c_C": 0.25,
            "c_D": 0.22,
        }

        vars = ["x1", "x2"]
        data = {}
        Ns = [1000, 1500]
        for st, var, N in zip(states, vars, Ns):
            data[var] = np.concatenate(
                [
                    np.random.normal(
                        loc=gt[f"mu_{s}"], scale=gt[f"sigma_{s}"], size=int(N * gt[f"c_{s}"]),
                    )
                    for s in st
                ]
            ).reshape(-1, 1)

        # todo guess is not used
        guess = {
            "mu_A": 0.2,
            "mu_B": 0.4,
            "mu_C": 0.7,
            "mu_D": 0.15,
            "sigma_A": 0.1,
            "sigma_B": 0.1,
            "sigma_C": 0.1,
            "sigma_D": 0.1,
            "c_A": 0.33,
            "c_B": 0.33,
            "c_C": 0.33,
            "c_D": 0.33,
        }

        model_dict = {}

        states = ["A", "B", "C"]
        g_shape = (1, 3)
        c_shape = (3, 1)
        mu = symbol_matrix(name="mu", shape=g_shape, suffix=states)
        sigma = symbol_matrix(name="sigma", shape=g_shape, suffix=states)
        c = symbol_matrix(name="c", shape=c_shape, suffix=states)
        model_dict[Symbol("p1")] = Mul(c, GMM(Symbol("x1"), mu, sigma))

        states = ["B", "C", "D"]
        mu = symbol_matrix(name="mu", shape=g_shape, suffix=states)
        sigma = symbol_matrix(name="sigma", shape=g_shape, suffix=states)
        c = symbol_matrix(name="c", shape=c_shape, suffix=states)
        model_dict[Symbol("p2")] = Mul(c, GMM(Symbol("x2"), mu, sigma))

        model = Model(model_dict)

        parameters = Parameters.from_symbols(model.symbols, gt)
        fit = Fit(model, parameters, data, loss=LogSumLoss(sum_axis=1))
        result = fit.execute(minimizer=LikelihoodOptimizer)

        expected = {
            "c_A": 0.2146783006888742,
            "c_B": 0.5418633676210778,
            "c_C": 0.24177790608861852,
            "c_D": 0.21747901002459014,
            "mu_B": 0.5528955016879634,
            "mu_A": 0.2296754306071527,
            "mu_D": 0.3386833210125317,
            "mu_C": 0.9223072648993988,
            "sigma_C": 0.093230280142199,
            "sigma_A": 0.0962655449677717,
            "sigma_B": 0.10288003410897235,
            "sigma_D": 0.21816387196646111,
        }

        for k in expected.keys():
            assert result.parameters[k] == pytest.approx(expected[k], rel=0.1)

    # @pytest.mark.skip("Long execution time")
    def test_markov_gmm(self):
        arr = np.genfromtxt(root_dir / "examples" / "data/GMM_dynamics.txt")
        data = {"e": arr[:, 0].reshape(-1, 1), "t": arr[:, 1]}

        guess_values = {
            "k_A_B": 1e-1,
            "k_B_A": 1e-1,
            "k_B_C": 1e-1,
            "y0_A": 0.6,
            "y0_B": 0.0,
            "mu_A": 0.7,
            "mu_B": 0.05,
            "mu_C": 0.4,
            "sigma_A": 0.1,
            "sigma_B": 0.2,
            "sigma_C": 0.1,
        }

        clear_symbols()
        # np.random.seed(43)

        connectivity = ["A <-> B -> C"]
        m = generate_transition_matrix(connectivity)
        states = extract_states(connectivity)

        # Temporal part
        xt = exp(m * Symbol("t"))

        # Future implementation needs constraints here, sum of y0 should be 1.
        y0 = Matrix([[Symbol("y0_A"), Symbol("y0_B"), 1 - Symbol("y0_A") - Symbol("y0_B")]]).T

        # Gaussian mixture model part
        mu = symbol_matrix("mu", shape=(1, 3), suffix=states)
        sigma = symbol_matrix("sigma", shape=(1, 3), suffix=states)
        gmm = GMM(Symbol("e"), mu=mu, sigma=sigma)

        model = Model({Symbol("p"): Mul(xt @ y0, gmm)})

        parameters = Parameters.from_symbols(model.symbols, guess_values)

        parameters.set("y0_A", lower_bound=0.0, upper_bound=1.0)
        parameters.set("y0_B", lower_bound=0.0, upper_bound=1.0, fixed=True)

        parameters.set("k_A_B", lower_bound=1e-3, upper_bound=1e2)
        parameters.set("k_B_A", lower_bound=1e-3, upper_bound=1e2)
        parameters.set("k_B_C", lower_bound=1e-3, upper_bound=1e2)

        # To calculate the likelihood for a measurement we need to sum the individual probabilities for all states
        # Thus we need to define which axis this is in the model
        STATE_AXIS = 1

        fit = Fit(model, parameters, data, loss=LogSumLoss(sum_axis=STATE_AXIS))
        result = fit.execute(minimizer=LikelihoodOptimizer, max_iter=200, verbose=True,)

        expected = {
            "k_A_B": 0.5415993464686054,
            "k_B_A": 0.08259132883479212,
            "k_B_C": 0.2527748185081457,
            "y0_A": 0.9696231022059791,
            "mu_A": 0.822262354106825,
            "mu_B": 0.12972476836918412,
            "mu_C": 0.5518311456516388,
            "sigma_A": 0.09320689716234987,
            "sigma_B": 0.12251906102401328,
            "sigma_C": 0.07922175330380453,
        }

        print(result.parameters)

        for k in expected:
            assert result.parameters[k] == pytest.approx(expected[k], rel=0.1)
