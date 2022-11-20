import pytest
from dont_fret.em_fit.datagen import generate_dataset
from sympy import HadamardProduct, Matrix, exp
import numpy as np

from slimfit.callable import convert_callable, GMM
from slimfit.fit import Fit
from slimfit.functions import gaussian, gaussian_sympy, gaussian_numpy
from slimfit.loss import LogSumLoss
from slimfit.markov import generate_transition_matrix, extract_states
from slimfit.minimizer import LikelihoodOptimizer
from slimfit.operations import Mul, MatMul
from slimfit.models import Model
from slimfit.symbols import (
    Variable,
    Probability,
    parameter_matrix,
    Parameter,
    clear_symbols,
)


class TestEMBase(object):
    def test_parameter_matrix(self):
        clear_symbols()
        m = parameter_matrix("A", shape=(3, 3))

        assert m.shape == (3, 3)
        elem = m[0, 0]
        assert elem.name == "A_0_0"
        assert elem.value == 1.0

        m = parameter_matrix(
            "A",
            values=[1, 2, 3],
            shape=(1, 3),
            suffix=["a", "b", None],
            vmin=[None, -3, -5],  # check for assigning vmin to not a parameter element?
            vmax=6.0,
            as_parameter=[[True, True, False]],
        )

        elem = m[0, 0]
        assert elem.name == "A_a"
        assert elem.vmin == None
        assert elem.vmax == 6.0

        elem = m[0, 1]
        assert elem.name == "A_b"
        assert elem.vmin == -3
        assert elem.vmax == 6.0

        elem = m[0, 2]
        assert elem == 3

        m = convert_callable(parameter_matrix("A", values=np.arange(9), shape=(3, 3)))
        for v, (i, j) in enumerate(np.ndindex(m.shape)):
            assert m.expr[i, j].value == v
            assert m[i, j].value == v
            assert m.kind == "constant"
            assert np.allclose(m(**m.guess), np.arange(9).reshape(3, 3))

        b = convert_callable(parameter_matrix("B", shape=(5, 2), rand_init=True, norm=True))
        assert b(**b.guess).sum() == pytest.approx(1.0)

    def test_model(self):
        clear_symbols()

        # Create GMM from sympy operations and Parameters
        mu = Matrix([Parameter("mu_1"), Parameter("mu_2"), Parameter("mu_3")])
        sigma = Matrix([Parameter("sigma_1"), Parameter("sigma_2"), Parameter("sigma_3")])

        g = gaussian(Variable("x"), mu, sigma)

        c = Matrix([Parameter("c_1"), Parameter("c_2"), Parameter("c_3")])
        model_dict = {Probability("p"): HadamardProduct(c, g)}

        model = Model(model_dict)
        rhs = next(iter(model.model_dict.values()))
        assert isinstance(rhs, Mul)

        # Test calling the model
        kwargs = {"x": np.linspace(0, 1, num=100), **model.guess}
        res = model(**kwargs)

        assert res["p"].shape == (100, 3, 1)

        # Create GMM with factory methods
        mu = parameter_matrix("mu", values=[0.2, 0.4, 0.7], shape=(3, 1))
        sigma = parameter_matrix("sigma", values=[0.1, 0.1, 0.1], shape=(3, 1))
        x = Variable("x")
        gmm = gaussian(x, mu, sigma)
        c = parameter_matrix("c", values=[0.2, 0.3, 0.5], shape=(3, 1))

        model_dict = {Probability("p"): Mul(c, gmm)}
        model = Model(model_dict)

        rhs = next(iter(model.model_dict.values()))
        assert isinstance(rhs, Mul)

        # Test calling the model
        kwargs = {"x": np.linspace(-0.5, 1.5, num=100), **model.guess}
        res = model(**kwargs)

        assert res["p"].shape == (100, 3, 1)

        # check integration to one

        val = sum(np.trapz(f, kwargs["x"]) for f in res["p"].squeeze().T)
        assert 1 == pytest.approx(val, 1)


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

        model = Model({Variable("y"): Parameter("a") * Variable("x") + Parameter("b")})
        fit = Fit(model, **data)

        res = fit.execute()

        assert res.parameters["a"] == pytest.approx(gt["a"], abs=0.2)
        assert res.parameters["b"] == pytest.approx(gt["b"], abs=0.1)

    def test_likelihood_gaussian(self):
        clear_symbols()
        np.random.seed(43)

        gt = {"mu": 2.4, "sigma": 0.7}

        xdata = np.random.normal(gt["mu"], scale=gt["sigma"], size=100)
        model = Model(
            {Probability("p"): gaussian_sympy(Variable("x"), Parameter("mu"), Parameter("sigma"))}
        )

        fit = Fit(model, x=xdata)
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

        x = parameter_matrix(name="X", shape=(3, 1))
        m = MatMul(basis, x)
        fit = Fit({Variable("b"): m}, b=spectrum)
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

        xt = exp(m * Variable("t"))
        y0 = parameter_matrix(name="y0", shape=(3, 1), suffix=states, rand_init=True, norm=True)
        model = Model({Variable("y"): xt @ y0})

        num = 50
        t = np.linspace(0, 11, num=num)
        populations = model(t=t, **gt_values)["y"]
        data = populations + np.random.normal(0, 0.05, size=num * 3).reshape(populations.shape)

        fit = Fit(model, y=data, t=t)
        result = fit.execute()

        expected = {
            "k_A_B": 1.0871957340661365,
            "k_B_A": 0.09590932036496363,
            "k_B_C": 0.5062539695609674,
            "y0_A": 1.02597395255225,
            "y0_B": -0.025642535216578468,
            "y0_C": -0.0022023742693679043,
        }

        for k in expected.keys():
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
            "sigma_D": 0.2,
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
        data = {"x": xdata}

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
            "c_D": 0.33,
        }

        mu = parameter_matrix(name="mu", shape=(3, 1), suffix=states, rand_init=True)
        sigma = parameter_matrix(name="sigma", shape=(3, 1), suffix=states, rand_init=True)
        c = parameter_matrix(name="c", shape=(3, 1), suffix=states, norm=True)
        model = Model({Probability("p"): Mul(c, GMM(Variable("x"), mu, sigma))})

        fit = Fit(model, **data)
        result = fit.execute(
            guess=guess, minimizer=LikelihoodOptimizer, loss=LogSumLoss(sum_axis=1), verbose=False
        )

        expected = {
            "c_A": 0.2180794404877423,
            "c_B": 0.5351105112590985,
            "c_C": 0.2468100482531592,
            "mu_A": 0.23155221598099562,
            "mu_B": 0.5508567564172897,
            "mu_C": 0.9204744537231175,
            "sigma_A": 0.09704934271877942,
            "sigma_B": 0.09910459765563108,
            "sigma_C": 0.09877267156818359,
        }
        for k in expected.keys():
            assert result.parameters[k] == pytest.approx(expected[k], rel=0.1)

        # repeat the fit with some of the parameters fixed
        Parameter("mu_A", value=0.2, fixed=True)
        Parameter("sigma_B", value=0.13, fixed=True)

        result = fit.execute(
            guess=guess, minimizer=LikelihoodOptimizer, loss=LogSumLoss(sum_axis=1), verbose=False
        )
        expected = {
            "c_A": 0.16735154876375555,
            "c_B": 0.6152426323837681,
            "c_C": 0.21740581885247645,
            "mu_B": 0.544103348133505,
            "mu_C": 0.9377787616082948,
            "sigma_A": 0.08280331849224268,
            "sigma_C": 0.08907335472319457,
        }
        for k in expected.keys():
            assert result.parameters[k] == pytest.approx(expected[k], rel=0.1)

        expected_fixed = {"mu_A": 0.2, "sigma_B": 0.13}

        for k in expected_fixed.keys():
            assert result.fixed_parameters[k] == pytest.approx(expected_fixed[k], rel=0.1)

    def test_global_gmm(self):
        """Test fitting of multiple GMM datasets with overlapping populations"""
        clear_symbols()
        np.random.seed(43)

        states = ["ABC", "BCD"]

        gt = {
            "mu_A": 0.23,
            "mu_B": 0.55,
            "mu_C": 0.92,
            "mu_D": 0.15,
            "sigma_A": 0.1,
            "sigma_B": 0.1,
            "sigma_C": 0.1,
            "sigma_D": 0.1,
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
            )

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
        mu = parameter_matrix(name="mu", shape=(3, 1), suffix=states, rand_init=True)
        sigma = parameter_matrix(name="sigma", shape=(3, 1), suffix=states, rand_init=True)
        c = parameter_matrix(name="c", shape=(3, 1), suffix=states, norm=True)
        model_dict[Probability("p1")] = Mul(c, GMM(Variable("x1"), mu, sigma))

        states = ["B", "C", "D"]
        mu = parameter_matrix(name="mu", shape=(3, 1), suffix=states, rand_init=True)
        sigma = parameter_matrix(name="sigma", shape=(3, 1), suffix=states, rand_init=True)
        c = parameter_matrix(name="c", shape=(3, 1), suffix=states, norm=True)
        model_dict[Probability("p2")] = Mul(c, GMM(Variable("x2"), mu, sigma))

        model = Model(model_dict)

        loss = LogSumLoss()
        opt = LikelihoodOptimizer(model, data, {}, loss=loss, guess=guess)
        result = opt.execute(max_iter=200, patience=5)

        expected = {
            "c_A": 0.22343642747136802,
            "c_B": 0.5292727366643867,
            "c_C": 0.24741157906292643,
            "c_D": 0.22323518880689952,
            "mu_C": 0.9195236095835075,
            "mu_A": 0.23544605174817337,
            "mu_B": 0.5556571686778503,
            "mu_D": 0.15689741329079615,
            "sigma_B": 0.09974632654431646,
            "sigma_D": 0.10460963367572337,
            "sigma_C": 0.09482194155204957,
            "sigma_A": 0.09952476600824725,
        }

        for k in expected.keys():
            assert result.parameters[k] == pytest.approx(expected[k], rel=0.1)

    @pytest.mark.skip("Long execution time")
    def test_markov_gmm(self):
        ds = generate_dataset("single_dynamic")
        clear_symbols()
        np.random.seed(43)

        connectivity = ["A <-> B -> C"]
        m = generate_transition_matrix(connectivity)
        states = extract_states(connectivity)

        # Temporal part
        xt = exp(m * Variable("t"))
        y0 = Matrix(
            [[Parameter("y0_A"), Parameter("y0_B"), 1 - Parameter("y0_A") - Parameter("y0_B"),]]
        ).T

        # Gaussian mixture model part
        mu = parameter_matrix("mu", shape=(3, 1), suffix=states)
        sigma = parameter_matrix("sigma", shape=(3, 1), suffix=states)
        gmm = GMM(Variable("e"), mu=mu, sigma=sigma)

        model = Model({Probability("p"): Mul(xt @ y0, gmm)})

        # Future implementation needs constraints here
        Parameter("y0_A", value=1.0, fixed=False, vmin=0.0, vmax=1.0)
        Parameter("y0_B", value=0.0, fixed=True, vmin=0.0, vmax=1.0)
        # y0_C is given by 1 - y0_A - y0_B

        Parameter("k_A_B", vmin=1e-3, vmax=1e2)
        Parameter("k_B_A", vmin=1e-3, vmax=1e2)
        Parameter("k_B_C", vmin=1e-3, vmax=1e2)

        STATE_AXIS = 1
        fit = Fit(model, **ds.data)
        result = fit.execute(
            guess=ds.guess,
            minimizer=LikelihoodOptimizer,
            max_iter=100,
            verbose=True,
            loss=LogSumLoss(sum_axis=STATE_AXIS),
        )

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

        for k in expected.keys():
            assert result.parameters[k] == pytest.approx(expected[k], rel=0.1)