# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Contains the ``QuantumMonteCarlo`` template and utility functions.
"""
import numpy as np

import pennylane as qml
from pennylane.templates.decorator import template
from pennylane.wires import Wires


def probs_to_unitary(probs):
    r"""Calculates the unitary matrix corresponding to an input probability distribution.

    For a given distribution :math:`p_{i}`, this function returns the unitary :math:`\mathcal{A}`
    that transforms the :math:`|0\rangle` state as

    .. math::

        \mathcal{A} |0\rangle = \sum_{i} \sqrt{p_{i}} |i\rangle,

    so that measuring the resulting state in the computational basis will give the state
    :math:`|i\rangle` with probability :math:`p_{i}`. Note that the returned unitary matrix is
    real and hence an orthogonal matrix.

    Args:
        probs (array): input probability distribution as a flat array

    Returns:
        array: unitary

    Raises:
        ValueError: if the input array is not flat or does not correspond to a probability
            distribution

    **Example:**

    >>> p = np.ones(4) / 4
    >>> probs_to_unitary(p)
    array([[ 0.5       ,  0.5       ,  0.5       ,  0.5       ],
           [ 0.5       , -0.83333333,  0.16666667,  0.16666667],
           [ 0.5       ,  0.16666667, -0.83333333,  0.16666667],
           [ 0.5       ,  0.16666667,  0.16666667, -0.83333333]])
    """
    if not np.allclose(sum(probs), 1) or min(probs) < 0:
        raise ValueError(
            "A valid probability distribution of non-negative numbers that sum to one"
            "must be input"
        )

    dim = len(probs)
    unitary = np.zeros((dim, dim))

    unitary[:, 0] = np.sqrt(probs)
    unitary = np.linalg.qr(unitary)[0]

    # The QR decomposition introduces a phase of -1. We remove this so that we are preparing
    # sqrt(p_{i}) rather than -sqrt(p_{i}). Even though both options are valid, it may be surprising
    # to prepare the negative version.
    unitary *= -1

    return unitary


def func_to_unitary(func, M):
    r"""Calculates the unitary that encodes a function onto an ancilla qubit register.

    Consider a function defined on a set of integers :math:`X = [0, 1, \ldots, M - 1]` whose output
    is bounded in the interval :math:`[0, 1]`, i.e., :math:`f: X \rightarrow [0, 1]`.

    This function returns a unitary :math:`\mathcal{R}` that performs the transformation:

    .. math::

        \mathcal{R} |i\rangle \otimes |0\rangle = |i\rangle\otimes \left(\sqrt{1 - f(i)}|0\rangle +
        \sqrt{f(i)} |1\rangle\right),

    In other words, for a given input state :math:`|i\rangle \otimes |0\rangle`, this unitary
    encodes the amplitude :math:`\sqrt{f(i)}` onto the :math:`|1\rangle` state of the ancilla qubit.
    Hence, measuring the ancilla qubit will result in the :math:`|1\rangle` state with probability
    :math:`f(i)`.

    Args:
        func (callable): A function defined on the set of integers :math:`X = [0, 1, \ldots, M - 1]`
            with output value inside :math:`[0, 1]`
        M (int): the number of integers that the function is defined on

    Returns:
        array: the :math:`\mathcal{R}` unitary

    Raises:
        ValueError: if func is not bounded with :math:`[0, 1]` for all :math:`X`

    **Example:**

    >>> func = lambda i: np.sin(i) ** 2
    >>> M = 16
    >>> func_to_unitary(func, M)
    array([[ 1.        ,  0.        ,  0.        , ...,  0.        ,
             0.        ,  0.        ],
           [ 0.        , -1.        ,  0.        , ...,  0.        ,
             0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.54030231, ...,  0.        ,
             0.        ,  0.        ],
           ...,
           [ 0.        ,  0.        ,  0.        , ..., -0.13673722,
             0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        , ...,  0.        ,
             0.75968791,  0.65028784],
           [ 0.        ,  0.        ,  0.        , ...,  0.        ,
             0.65028784, -0.75968791]])
    """
    unitary = np.zeros((2 * M, 2 * M))

    for i in range(M):
        f = func(i)

        if not 0 <= f <= 1:
            raise ValueError(
                "func must be bounded within the interval [0, 1] for the range of" "input values"
            )

        unitary[2 * i, 2 * i] = np.sqrt(1 - f)
        unitary[2 * i + 1, 2 * i] = np.sqrt(f)
        unitary[2 * i, 2 * i + 1] = np.sqrt(f)
        unitary[2 * i + 1, 2 * i + 1] = -np.sqrt(1 - f)

    return unitary


def _make_V(dim):
    r"""Calculates the :math:`\mathcal{V}` unitary which performs a reflection along the
    :math:`|1\rangle` state of the end ancilla qubit.

    Args:
        dim (int): dimension of :math:`\mathcal{V}`

    Returns:
        array: the :math:`\mathcal{V}` unitary
    """
    assert dim % 2 == 0, "dimension for _make_V() must be even"

    one = np.array([[0, 0], [0, 1]])
    dim_without_qubit = int(dim / 2)

    return 2 * np.kron(np.eye(dim_without_qubit), one) - np.eye(dim)


def _make_Z(dim):
    r"""Calculates the :math:`\mathcal{Z}` unitary which performs a reflection along the all
    :math:`|0\rangle` state.

    Args:
        dim (int): dimension of :math:`\mathcal{Z}`

    Returns:
        array: the :math:`\mathcal{Z}` unitary
    """
    Z = -np.eye(dim)
    Z[0, 0] = 1
    return Z


def make_Q(A, R):
    """Calculates the :math:`\mathcal{Q}` matrix that encodes the expectation value according to the
    probability unitary :math:`\mathcal{A}` and the random variable unitary :math:`\mathcal{R}`.

    Following `this <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.98.022321>`__ paper,
    the expectation value is encoded as the phase of an eigenvalue of :math:`\mathcal{Q}`. This
    phase can be estimated using quantum phase estimation, see :func:`~.QuantumPhaseEstimation`.

    Args:
        A (array): The unitary matrix of :math:`\mathcal{A}` which encodes the probability
            distribution
        R (array): The unitary matrix of :math:`\mathcal{R}` which encodes the function

    Returns:
        array: the :math:`\mathcal{Q}` unitary
    """
    A_big = np.kron(A, np.eye(2))
    F = R @ A_big
    F_dagger = F.conj().T

    dim = len(R)
    V = _make_V(dim)
    Z = _make_Z(dim)
    UV = F @ Z @ F_dagger @ V

    return UV @ UV


@template
def QuantumMonteCarlo(probs, func, target_wires, estimation_wires):
    """TODO"""
    if isinstance(probs, np.ndarray) and probs.ndim != 1:
        raise ValueError("The probability distribution must be specified as a flat array")

    dim_p = len(probs)
    num_target_wires = np.log2(2 * dim_p)

    if not num_target_wires.isinteger():
        raise ValueError("The probability distribution must have a length that is a power of two")

    num_target_wires = int(num_target_wires)

    target_wires = Wires(target_wires)
    estimation_wires = Wires(estimation_wires)

    if num_target_wires != len(target_wires):
        raise ValueError(f"The probability distribution of dimension {dim_p} requires"
                         f" {num_target_wires} target wires")

    A = probs_to_unitary(probs)
    R = func_to_unitary(func, dim_p)
    Q = make_Q(A, R)

    qml.QubitUnitary(A, wires=target_wires[:-1])
    qml.QubitUnitary(R, wires=target_wires)
    qml.templates.QuantumPhaseEstimation(Q, target_wires=target_wires, estimation_wires=estimation_wires)
