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
Contains the ``QuantumMonteCarlo`` template.
"""
import itertools

import numpy as np

from pennylane.templates.decorator import template


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
    if isinstance(probs, np.ndarray) and probs.ndim != 1:
        raise ValueError("The probability distribution must be specified as a flat array")
    if not np.allclose(sum(probs), 1) or min(probs) < 0:
        raise ValueError("A valid probability distribution of non-negative numbers that sum to one"
                         "must be input")

    dim = len(probs)
    unitary = np.zeros((dim, dim))

    unitary[:, 0] = np.sqrt(probs)
    unitary = np.linalg.qr(unitary)[0]

    # The QR decomposition introduces a phase of -1. We remove this so that we are preparing
    # sqrt(p_{i}) rather than -sqrt(p_{i}). Even though both options are valid, it may be surprising
    # to prepare the negative version.
    unitary *= -1

    return unitary


def func_to_unitary(xs, func):
    r"""Calculates the unitary that encodes a function onto an ancilla qubit register.

    Consider the set of :math:`M` points :math:`X = \{x_{0}, x_{1}, \ldots, x_{M - 1}\}` and the
    corresponding :math:`d`-dimensional discrete space :math:`X^{d}`. Also, consider the function
    :math:`f: X^{d} \rightarrow \mathbb{R}` such that :math:`0 \leq f(x) \leq 1` for all
    :math:`x \in X`.

    This function returns a unitary :math:`\mathcal{R}` that performs the transformation:

    .. math::

        \mathcal{R} |0\rangle \otimes |i_{d - 1}i_{d - 2}\ldots i_{0}\rangle
         = \left(\sqrt{1 - f(x_{i_{0}}, \ldots, x_{i_{d - 2}}, x_{i_{d - 1}})}|0\rangle +
        \sqrt{f(x_{i_{0}}, \ldots, x_{i_{d - 2}}, x_{i_{d - 1}})} |1\rangle\right)
        \otimes |i_{d - 1}i_{d - 2}\ldots i_{0}\rangle,

    where :math:`i_{j} \in \{0, 1, \ldots , M - 1\}` for all :math:`j`. For a given input state
    :math:`|0\rangle \otimes |i_{d - 1}i_{d - 2}\ldots i_{0}\rangle`, this unitary encodes the
    amplitude :math:`\sqrt{f(x_{i_{0}}, \ldots, x_{i_{d - 2}}, x_{i_{d - 1}})}` onto the
    :math:`|1\rangle` state of the ancilla qubit. Hence, measuring the ancilla qubit will result
    in the :math:`|1\rangle` state with probability
    :math:`f(x_{i_{0}}, \ldots, x_{i_{d - 2}}, x_{i_{d - 1}})`.

    More generally, one can consider a :math:`d`-dimensional discrete product space
    :math:`X_{0} \times X_{1} \times \ldots \times X_{d - 1}` with each :math:`X_{i}` composed of
    an independent set of :math:`x`-values of varying length.

    Args:
        xs (list[array]): a list of arrays containing the values for each :math:`X_{i}`
        func (callable): A random variable that can be called with :math:`d` arguments.
            It must evaluate within :math:`[0, 1]` for the range of input values specified by
            ``xs``.

    Returns:
        array: the :math:`\mathcal{R}` unitary

    Raises:
        ValueError: if func is not bounded with :math:`[0, 1]` for the range of input values
            specified by ``xs``

    **Example:**

    >>> M = 8
    >>> x = np.linspace(-np.pi, np.pi, M)
    >>> d = 2
    >>> xs = [x] * d
    >>> func = lambda x1, x2: np.sin(x1 - x2) ** 2
    >>> func_to_unitary(xs, func)
    array([[ 1.        ,  0.        ,  0.        , ...,  0.        ,
             0.        ,  0.        ],
           [ 0.        ,  0.6234898 ,  0.        , ...,  0.        ,
             0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.22252093, ...,  0.        ,
             0.        ,  0.        ],
           ...,
           [ 0.        ,  0.        ,  0.        , ..., -0.22252093,
             0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        , ...,  0.        ,
            -0.6234898 ,  0.        ],
           [ 0.        ,  0.        ,  0.        , ...,  0.        ,
             0.        , -1.        ]])
    """
    dim = np.prod([len(x) for x in xs])
    unitary = np.zeros((2 * dim, 2 * dim))

    for i, args in enumerate(itertools.product(*reversed(xs))):
        f = func(*args)

        if not 0 <= f <= 1:
            raise ValueError("func must be bounded within the interval [0, 1] for the range of"
                             "input values")

        unitary[i, i] = np.sqrt(1 - f)
        unitary[i + dim, i] = np.sqrt(f)
        unitary[i, i + dim] = np.sqrt(f)
        unitary[i + dim, i + dim] = - np.sqrt(1 - f)

    return unitary




@template
def QuantumMonteCarlo(distributions, random_variable, target_wires, estimation_wires):
    """TODO
    """
    ...
