import numpy as np
import pytest

import pennylane as qml
from pennylane import qchem

import shutil


delta_h2 = 0.00529
x_h2 = np.array([0.0, 0.0, -0.332568, 0.0, 0.0, -1.067432])
hessian_h2 = np.array(
    [
        [-2.77382489e-17, 7.69895020e-19, -2.70561764e-17],
        [2.41555458e-18, 8.16338574e-01, 3.18442967e-03],
        [-3.88375990e-17, 3.18442967e-03, 4.00839669e-02],
    ]
)
params_h2 = np.array([2.98479079, -0.22344263, -0.01805065])

deriv_h2_00 = 1.158703853412768e-05
deriv_h2_34 = 0.0
deriv_h2_22 = 0.4771353388653741
deriv_h2_25 = -0.47713533888071563


def H2(x):
    return qml.qchem.molecular_hamiltonian(["H", "H"], x)[0]


@pytest.mark.parametrize(
    ("H", "x", "i", "j", "params", "hessian", "delta", "exp_deriv"),
    [(H2, x_h2, 0, 0, params_h2, hessian_h2, delta_h2, deriv_h2_00),
     (H2, x_h2, 3, 4, params_h2, hessian_h2, delta_h2, deriv_h2_34),
     (H2, x_h2, 2, 2, params_h2, hessian_h2, delta_h2, deriv_h2_22),
     (H2, x_h2, 2, 5, params_h2, hessian_h2, delta_h2, deriv_h2_25)],
)
def test_deriv2_energy(H, x, i, j, params, hessian, delta, exp_deriv, tol):
    r"""Tests the ``second_derivative_energy`` function computing the second-order derivative
    of the total energy ``E(x)`` at the nuclear coordinates ``x``.
    """

    dev = qml.device("default.qubit", wires=4)

    def circuit(params, wires):
        qml.BasisState(np.array([1, 1, 0, 0]), wires=[0, 1, 2, 3])
        qml.Rot(*params, wires=2)
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[2, 0])
        qml.CNOT(wires=[3, 1])

    deriv2_energy = qchem.second_derivative_energy(
        H, x, i, j, circuit, params, dev, hessian, delta=delta
    )

    shutil.rmtree("pyscf")

    assert np.allclose(deriv2_energy, exp_deriv, **tol)
