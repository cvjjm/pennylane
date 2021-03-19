# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
This module contains the mixin interface class for creating differentiable quantum tapes with
Autograd.
"""
# pylint: disable=protected-access
import autograd.extend
import autograd.builtins
from autograd.numpy.numpy_boxes import ArrayBox, Box

from pennylane import numpy as np
from pennylane.tape.queuing import AnnotatedQueue


class AutogradInterface(AnnotatedQueue):
    """Mixin class for applying an autograd interface to a :class:`~.JacobianTape`.

    Autograd-compatible quantum tape classes can be created via subclassing:

    .. code-block:: python

        class MyAutogradQuantumTape(AutogradInterface, JacobianTape):

    Alternatively, the autograd interface can be dynamically applied to existing
    quantum tapes via the :meth:`~.apply` class method. This modifies the
    tape **in place**.

    Once created, the autograd interface can be used to perform quantum-classical
    differentiable programming.

    .. note::

        If using a device that supports native autograd computation and backpropagation, such as
        :class:`~.DefaultQubitAutograd`, the Autograd interface **does not need to be applied**. It
        is only applied to tapes executed on non-Autograd compatible devices.

    **Example**

    Once an autograd quantum tape has been created, it can be differentiated using autograd:

    .. code-block:: python

        tape = AutogradInterface.apply(JacobianTape())

        with tape:
            qml.Rot(0, 0, 0, wires=0)
            expval(qml.PauliX(0))

        def cost_fn(x, y, z, device):
            tape.set_parameters([x, y ** 2, y * np.sin(z)], trainable_only=False)
            return tape.execute(device=device)

    >>> x = np.array(0.1, requires_grad=False)
    >>> y = np.array(0.2, requires_grad=True)
    >>> z = np.array(0.3, requires_grad=True)
    >>> dev = qml.device("default.qubit", wires=2)
    >>> cost_fn(x, y, z, device=dev)
    [0.03991951]
    >>> jac_fn = qml.jacobian(cost_fn)
    >>> jac_fn(x, y, z, device=dev)
    [[ 0.39828408, -0.00045133]]
    """

    # pylint: disable=attribute-defined-outside-init
    dtype = np.float64

    @property
    def interface(self):  # pylint: disable=missing-function-docstring
        return "autograd"

    def _update_trainable_params(self):
        """Set the trainable parameters.

        Unlike in :class:`~.JacobianTape`, we also set the private attribute
        ``self._all_parameter_values``.
        """
        params = self.get_parameters(trainable_only=False, return_arraybox=True)
        trainable_params = set()

        for idx, p in enumerate(params):
            if getattr(p, "requires_grad", False) or isinstance(p, ArrayBox):
                trainable_params.add(idx)

        self.trainable_params = trainable_params
        self._all_parameter_values = params

    def get_parameters(self, trainable_only=True, return_arraybox=False):
        """Return the parameters incident on the tape operations.

        The returned parameters are provided in order of appearance
        on the tape. By default, the returned parameters are wrapped in
        an ``autograd.builtins.list`` container.

        Args:
            trainable_only (bool): if True, returns only trainable parameters
            return_arraybox (bool): if True, the returned parameters are not
                wrapped in an ``autograd.builtins.list`` container
        Returns:
            autograd.builtins.list or list: the corresponding parameter values

        **Example**

        .. code-block:: python

            with JacobianTape() as tape:
                qml.RX(0.432, wires=0)
                qml.RY(0.543, wires=0)
                qml.CNOT(wires=[0, 'a'])
                qml.RX(0.133, wires='a')
                expval(qml.PauliZ(wires=[0]))

        By default, all parameters are trainable and will be returned:

        >>> tape.get_parameters()
        [0.432, 0.543, 0.133]

        Setting the trainable parameter indices will result in only the specified
        parameters being returned:

        >>> tape.trainable_params = {1} # set the second parameter as free
        >>> tape.get_parameters()
        [0.543]

        The ``trainable_only`` argument can be set to ``False`` to instead return
        all parameters:

        >>> tape.get_parameters(trainable_only=False)
        [0.432, 0.543, 0.133]
        """
        params = []
        iterator = self.trainable_params if trainable_only else self._par_info

        for p_idx in iterator:
            op = self._par_info[p_idx]["op"]
            op_idx = self._par_info[p_idx]["p_idx"]
            params.append(op.data[op_idx])

        return params if return_arraybox else autograd.builtins.list(params)

    def _execute(self, params, device):
        """Wrapper to cath the needless autograd forward pass

        This function is not primitive to autograd so when params
        is a autograd box we know that the following call to
        the primitve wrapped function is the autograd forward pass.
        """
        if isinstance(params, autograd.extend.Box):
            self.delay_execute_device_as_this_is_a_potentially_needless_autograd_forward_pass = True
        else:
            self.delay_execute_device_as_this_is_a_potentially_needless_autograd_forward_pass = False
        return self.actual_execute(params, device)

    @autograd.extend.primitive
    def actual_execute(self, params, device):

        # unwrap constant parameters
        self._all_params_unwrapped = [
            p.numpy() if isinstance(p, np.tensor) else p for p in self._all_parameter_values
        ]

        if self.delay_execute_device_as_this_is_a_potentially_needless_autograd_forward_pass:
            self.delay_execute_device_as_this_is_a_potentially_needless_autograd_forward_pass = False

            class _DelayedExecutionTensor(np.tensor):
                """A numpy tensor that delays computation of its content until it is actually needed
                """
                def bind(self, *, autograd_interface=None, params=None, device=None, output_dim=None):
                    """Bind this numpy tensor to the current interface instance for delayed execution
                    """
                    #print("bind()")
                    self.autograd_interface = autograd_interface
                    self.params = params
                    self.device = device
                    self.output_dim = output_dim
                    self.res = None

                def execute(self):
                    #print("execute")
                    if hasattr(self, 'res') and self.res is None:
                        print("* * * executig * * *")
                        print(self.shape)
                        res = self.autograd_interface.actual_execute(self.params, self.device)
                        self.res = res
                        # Now we want self to look like res
                        #self.data = res.data # option 1 (may lead to problems with incompatible dtypes)
                        np.copyto(self, res) # option 2 (may lead to problems with incompatible dtypes)
                        print(self.shape, res.shape)
                        # self.__dict__ = res.__dict__ # option 3 (does not work)

                @property
                def dtype(self):
                    if hasattr(self, 'res') and self.res is None:
                        return res.dtype
                    return super().dtype

                def unwrap(self):
                    self.execute()
                    return super().unwrap()

                def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                    self.execute()
                    return super().__array_ufunc__(ufunc, method, *inputs, **kwargs)

                @property
                def ndim(self):
                    self.execute()
                    return self.res.ndim

                # def __array_function__(self, func, types, args, kwargs):
                #     print("__array_function__")
                #     return NotImplemented

                def squeeze(self, a, axis=None):
                    self.execute()
                    return super().squeeze(self, a, axis=axis)

                def __array_finalize__(self, obj):
                    print("__array_finalize__(", type(self), type(obj), ")")
                    self.execute()
                    super().__array_finalize__(obj)

                # @property
                # def shape(self):
                #     print("shape", super().shape)
                #     #return (self.output_dim,)
                #     return super().shape

                # @property
                # def dtype(self):
                #     print("dtype", super().dtype)
                #     #return float
                #     return super().dtype

                # def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                #     """Intercept universal functions on this tensor and do the execution

                #     This is never called durinng a
                #     """
                #     print("__array_ufunc__", ufunc, method, inputs, kwargs)
                #     # use the bound interface to do the execution now
                #     self.execute()
                #     # inject the actual result into the ufunc inputs
                #     inputs_with_result_injected = tuple([self.res if isinstance(inp, self.__class__) else inp for inp in inputs])
                #     # return ufunc applied to resul
                #     array_ufunc_return = self.res.__array_ufunc__(ufunc, method, *inputs_with_result_injected, **kwargs)
                #     # replace
                #     #self.__dict__.update(self.res.__dict__)
                #     #print("array_ufunc_return", type(array_ufunc_return))
                #     return array_ufunc_return


            # class Meta(type):
            #     def __getattribute__(*args):
            #         print("Metaclass __getattribute__", args)
            #         return type.__getattribute__(*args)

            class DelayedExecutionTensor(_DelayedExecutionTensor):
                # __metaclass__ = Meta
                def __getattribute__(self, name):
                    print("__getattribute__(", name, ")", )
                    if hasattr(super(), 'res') and super().res is not None:
                        return super().res.__getattribute__(name)
                    return super().__getattribute__(name)

            # register the DelayedExecutionTensor with autograd
            ArrayBox.register(DelayedExecutionTensor)
            autograd.extend.VSpace.register(DelayedExecutionTensor, lambda x: autograd.numpy.numpy_vspaces.ArrayVSpace(x))

            # make a DelayedExecutionTensor with suitable shape,
            fake_res = DelayedExecutionTensor(np.zeros(self.output_dim))

            fake_res.bind(autograd_interface=self, params=params, device=device, output_dim=self.output_dim)
            return fake_res

        # unwrap all NumPy scalar arrays to Python literals
        params = [p.item() if p.shape == tuple() else p for p in params]
        params = autograd.builtins.tuple(params)

        # evaluate the tape
        self.set_parameters(self._all_params_unwrapped, trainable_only=False)
        res = self.execute_device(params, device=device)
        self.set_parameters(self._all_parameter_values, trainable_only=False)

        if self.is_sampled:
            return res

        if res.dtype == np.dtype("object"):
            return np.hstack(res)

        requires_grad = False

        if self.trainable_params:
            requires_grad = True

        return np.array(res, requires_grad=requires_grad)

    @staticmethod
    def vjp(ans, self, params, device):  # pylint: disable=unused-argument
        """Returns the vector-Jacobian product operator for the quantum tape.
        The returned function takes the arguments as :meth:`~.JacobianTape.execute`.

        Args:
            ans (array): the result of the tape execution
            self (.AutogradQuantumTape): the tape instance
            params (list[Any]): the quantum tape operation parameters
            device (.Device): a PennyLane device that can execute quantum
                operations and return measurement statistics

        Returns:
            function: this function accepts the backpropagation
            gradient output vector, and computes the vector-Jacobian product
        """

        saved_grad_matrices = {}

        def _evaluate_grad_matrix(p, grad_matrix_fn):
            """Convenience function for generating gradient matrices
            for the given parameter values.

            This function serves two purposes:

            * Avoids duplicating logic surrounding parameter unwrapping/wrapping

            * Takes advantage of closure, to cache computed gradient matrices via
              the ``saved_grad_matrices`` attribute, to avoid gradient matrices being
              computed multiple redundant times.

              This is particularly useful when differentiating vector-valued QNodes.
              Because Autograd requests the vector-grad matrix product,
              and *not* the full grad matrix, differentiating vector-valued
              functions will result in multiple backward passes.
            """
            if grad_matrix_fn in saved_grad_matrices:
                return saved_grad_matrices[grad_matrix_fn]

            self.set_parameters(self._all_params_unwrapped, trainable_only=False)
            grad_matrix = getattr(self, grad_matrix_fn)(device, params=p, **self.jacobian_options)
            self.set_parameters(self._all_parameter_values, trainable_only=False)

            saved_grad_matrices[grad_matrix_fn] = grad_matrix
            return grad_matrix

        def gradient_product(dy):
            """Returns the vector-Jacobian product with given
            parameter values p and output gradient dy"""

            @autograd.extend.primitive
            def jacobian(p):
                """Returns the Jacobian for parameters p"""
                return _evaluate_grad_matrix(p, "jacobian")

            def vhp(ans, p):
                def hessian_product(ddy):
                    """Returns the vector-Hessian product with given
                    parameter values p, output gradient dy, and output
                    second-order gradient ddy"""
                    hessian = _evaluate_grad_matrix(p, "hessian")

                    if dy.size > 1:
                        if all(np.ndim(p) == 0 for p in params):
                            # only flatten dy if all parameters are single values
                            vhp = dy.flatten() @ ddy @ hessian @ dy.flatten()
                        else:
                            vhp = dy @ ddy @ hessian @ dy.T
                    else:
                        vhp = np.squeeze(ddy @ hessian)

                    return vhp

                return hessian_product

            # register vhp as the backward method of the jacobian function
            autograd.extend.defvjp(jacobian, vhp, argnums=[0])

            # only flatten dy if all parameters are single values
            if all(np.ndim(p) == 0 for p in params):
                vjp = dy.flatten() @ jacobian(params)
            else:
                vjp = dy @ jacobian(params)

            return vjp

        return gradient_product

    @classmethod
    def apply(cls, tape):
        """Apply the autograd interface to an existing tape in-place.

        Args:
            tape (.JacobianTape): a quantum tape to apply the Autograd interface to

        **Example**

        >>> with JacobianTape() as tape:
        ...     qml.RX(0.5, wires=0)
        ...     expval(qml.PauliZ(0))
        >>> AutogradInterface.apply(tape)
        >>> tape
        <AutogradQuantumTape: wires=<Wires = [0]>, params=1>
        """
        tape_class = getattr(tape, "__bare__", tape.__class__)
        tape.__bare__ = tape_class
        tape.__class__ = type("AutogradQuantumTape", (cls, tape_class), {})
        tape._update_trainable_params()
        return tape


autograd.extend.defvjp(AutogradInterface.actual_execute, AutogradInterface.vjp, argnums=[1])
