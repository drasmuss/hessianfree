"""Implementation of Hessian-free optimization for recurrent networks.

Author: Daniel Rasmussen (drasmussen@princeton.edu)

Based on
Martens, J., & Sutskever, I. (2011). Learning recurrent neural networks with
hessian-free optimization. Proceedings of the 28th International Conference on
Machine Learning.
"""

import numpy as np

from hessianff import HessianFF
from nonlinearities import Continuous


class HessianRNN(HessianFF):
    def __init__(self, struc_damping=0.0, **kwargs):
        super(HessianRNN, self).__init__(**kwargs)
        # TODO: allow user to specify which layers are recurrent, instead
        # of assuming all except first/last

        self.struc_damping = struc_damping

        # add on recurrent weights
        self.W = np.concatenate(
            (self.W, self.init_weights([(l + 1, l)
                                        for l in self.shape[1:-1]],
                                       coeff=self.W_init_coeff)))

    def compute_offsets(self):
        """Precompute offsets for layers in the overall parameter vector."""

        super(HessianRNN, self).compute_offsets()

        self.rec_offsets = {}

        for l in range(1, self.n_layers - 1):
            offset = (len(self.W) +  # note: gets called before rec_W added
                      np.sum([(self.shape[i] + 1) * self.shape[i]
                              for i in range(1, l)]))
            self.rec_offsets[l] = (
                offset,
                offset + self.shape[l] * self.shape[l],
                offset + (self.shape[l] + 1) * self.shape[l])

    def get_weights(self, params, layer, separate=True, recurrent=False):
        """Get weight matrix for a layer from the overall parameter vector."""

        if not recurrent:
            return super(HessianRNN, self).get_weights(params, layer, separate)
        else:
            if layer not in self.rec_offsets:
                return None
            offset, W_end, b_end = self.rec_offsets[layer]
            if separate:
                W = params[offset:W_end]
                b = params[W_end:b_end]

                return (W.reshape((self.shape[layer],
                                   self.shape[layer])),
                        b)
            else:
                return params[offset:b_end].reshape((self.shape[layer] + 1,
                                                     self.shape[layer]))

    def forward(self, input, params, deriv=False):
        """Compute activations for given input sequence and parameters."""

        # input shape = [batch_size, seq_len, input_dim]
        # activations shape = [n_layers, seq_len, batch_size, layer_size]
        # note: seq_len and batch_size are swapped; this may be a bit confusing
        # but it makes the indexing a lot nicer in the rest of the code

        if input.ndim < 3:
            # then we've just been given a single sample (rather than batch)
            input = input[None, :, :]

        for l in self.layer_types:
            l.reset()

        activations = [np.zeros((input.shape[1], input.shape[0], l),
                                dtype=self.dtype)
                       for l in self.shape]

        if deriv:
            d_activations = [np.zeros_like(activations[i])
                             for i in range(self.n_layers)]

        W_recs = [self.get_weights(params, i, recurrent=True)
                  for i in np.arange(self.n_layers)]
        for s in range(input.shape[1]):
            # input activations
            activations[0][s] = self.act[0](input[:, s])

            if deriv:
                d_activations[0][s] = self.deriv[0](
                    activations[0][s] if self.layer_types[0].use_activations
                    else input[:, s])

            for i in range(1, self.n_layers):
                # feedforward input
                ff_input = np.zeros_like(activations[i][s])
                for pre in self.back_conns[i]:
                    W, b = self.get_weights(params, (pre, i))
                    ff_input += np.dot(activations[pre][s], W) + b

                if i == self.n_layers - 1:
                    # no recurrent connections for last layer
                    rec_input = 0
                else:
                    if s > 0:
                        rec_input = np.dot(activations[i][s - 1],
                                           W_recs[i][0])
                    else:
                        # bias input on first timestep
                        rec_input = W_recs[i][1]

                activations[i][s] = self.act[i](ff_input + rec_input)

                if deriv:
                    d_activations[i][s] = self.deriv[i](
                        activations[i][s] if
                        self.layer_types[i].use_activations
                        else ff_input + rec_input)

        if deriv:
            return activations, d_activations

        return activations

    def error(self, W=None, inputs=None, targets=None):
        """Compute network error."""

        W = self.W if W is None else W
        inputs = self.inputs if inputs is None else inputs
        targets = self.targets if targets is None else targets
        targets = np.swapaxes(targets, 0, 1)

        return super(HessianRNN, self).error(W, inputs, targets)

    def calc_grad(self, W=None, inputs=None, targets=None):
        """Compute parameter gradient."""

        W = self.W if W is None else W
        inputs = self.inputs if inputs is None else inputs
        targets = self.targets if targets is None else targets

        if W is self.W and inputs is self.inputs:
            # use cached activations
            activations = self.activations
            GPU_activations = self.GPU_activations
            d_activations = self.d_activations
        else:
            # compute activations
            activations, d_activations = self.forward(input, W, deriv=True)
            GPU_activations = None

        grad = np.zeros(W.size, dtype=self.dtype)
        deltas = [np.zeros((inputs.shape[0], l), dtype=self.dtype)
                  for l in self.shape]
        W_rec = [self.get_weights(W, l, recurrent=True)
                 for l in np.arange(self.n_layers - 1)]

        # backpropagate error
        for s in range(inputs.shape[1] - 1, -1, -1):
            if self.error_type == "mse":
                error = activations[-1][s] - np.nan_to_num(targets[:, s])
            elif self.error_type == "ce":
                error = -np.nan_to_num(targets[:, s]) / activations[-1][s]

            if self.layer_types[-1].d_state is None:
                deltas[-1] = self.J_dot(d_activations[-1][s], error)
            else:
                deltas[-1] = (self.J_dot(d_activations[-1][s], error) +
                              self.layer_types[-1].d_state * deltas[-1])


            for l in range(self.n_layers - 2, -1, -1):
                # gradient for feedforward weights
                error = np.zeros_like(deltas[l])
                for post in self.conns[l]:
                    c_error = np.dot(deltas[post],
                                     self.get_weights(W, (l, post))[0].T)
                    error += c_error
                    offset, W_end, b_end = self.offsets[(l, post)]
                    grad[offset:W_end] += (
                        self.outer_sum(activations[l][s]
                                       if GPU_activations is None
                                       else [l, s],
                                       deltas[post]))
                    grad[W_end:b_end] += np.sum(deltas[post], axis=0)

                if l > 0:
                    # gradient for recurrent weights
                    error += np.dot(deltas[l], W_rec[l][0].T)
                    if self.layer_types[l].d_state is None:
                        deltas[l] = self.J_dot(d_activations[l][s], error)
                    else:
                        deltas[l] = (self.J_dot(d_activations[l][s], error) +
                            self.layer_types[l].d_state * deltas[l])


                    offset, W_end, b_end = self.rec_offsets[l]
                    if s > 0:
                        grad[offset:W_end] += (
                            self.outer_sum(activations[l][s - 1]
                                           if GPU_activations is None
                                           else [l, s - 1],
                                           deltas[l]))
                    else:
                        # put remaining gradient into initial bias
                        grad[W_end:b_end] = np.sum(deltas[l], axis=0)

        # divide by batchsize
        grad /= inputs.shape[0]

        return grad

    def check_G(self, calc_G, inputs, targets, v, damping=0):
        """Check Gv calculation via finite differences (for debugging)."""

        # TODO: get struc_damping check to work
        assert self.struc_damping == 0

        eps = 1e-6
        N = self.W.size
        sig_len = inputs.shape[1]

        g = np.zeros(N)
        for b in range(inputs.shape[0]):
            acts = self.forward(inputs[b], self.W)

            # check_G only works for 1D output at the moment
            assert acts[-1].shape[2] == 1

            base = acts[-1].squeeze(axis=(1, 2))
#             base = np.concatenate((acts[-1].squeeze(axis=(1, 2)),
#                                    acts[1][-1].squeeze(axis=0)))

            J = np.zeros((sig_len, N))
#             J = np.zeros((sig_len + self.shape[1], N))
            for i in range(N):
                inc_i = np.zeros(N)
                inc_i[i] = eps

                acts = self.forward(inputs[b], self.W + inc_i)
                J[:, i] = (acts[-1].squeeze(axis=(1, 2)) - base) / eps
#                 J[:, i] = (np.concatenate((acts[-1].squeeze(axis=(1, 2)),
#                                            acts[1][-1].squeeze(axis=0))) -
#                            base) / eps

            if self.error_type == "mse":
                L = np.ones(sig_len)
            elif self.error_type == "ce":
                L = targets[b].squeeze(axis=1) / base ** 2

            L = np.diag(L)
#             L = np.diag(np.concatenate((L,
#                                         [self.struc_damping] *
#                                         self.shape[1])))

            g += np.dot(np.dot(J.T, np.dot(L, J)), v)

        g /= inputs.shape[0]

        g += damping * v

        try:
            assert np.allclose(g, calc_G, rtol=1e-3)
        except AssertionError:
            print g
            print calc_G
            print g - calc_G
            print g / calc_G
            raise

    def G(self, v, damping=0, output=None):
        """Compute Gauss-Newton matrix-vector product."""

        if output is None:
            Gv = np.zeros(self.W.size, dtype=self.dtype)
        else:
            Gv = output
            Gv[:] = 0

        sig_len = self.inputs.shape[1]

        # R forward pass
        R_inputs = [np.zeros(self.activations[i].shape, dtype=self.dtype)
                    for i in np.arange(self.n_layers)]
        R_activations = [None for _ in self.shape]
        R_outputs = np.zeros_like(self.activations[-1])
        v_recs = [self.get_weights(v, l, recurrent=True)
                  for l in np.arange(self.n_layers - 1)]
        W_recs = [self.get_weights(self.W, l, recurrent=True)
                  for l in np.arange(self.n_layers - 1)]

        R_activations[0] = np.zeros((self.inputs.shape[0], self.shape[0]),
                                    dtype=self.dtype)
        for s in np.arange(sig_len):
            for l in np.arange(1, self.n_layers):
                for pre in self.back_conns[l]:
                    vw, vb = self.get_weights(v, (pre, l))
                    Ww, _ = self.get_weights(self.W, (pre, l))
                    R_inputs[l][s] += np.dot(self.activations[pre][s],
                                             vw) + vb
                    R_inputs[l][s] += np.dot(R_activations[pre], Ww)

                if isinstance(self.layer_types[l], Continuous):
                    # TODO: general case for this (like d_state)?
                    R_inputs[l][s] += (R_inputs[l][s - 1] *
                                       self.layer_types[l].coeff)

                if l < self.n_layers - 1:
                    # add recurrent input
                    if s == 0:
                        R_inputs[l][s] += v_recs[l][1]
                    else:
                        R_inputs[l][s] += (
                            np.dot(self.activations[l][s - 1],
                                   v_recs[l][0]) +
                            np.dot(R_activations[l], W_recs[l][0]))

                R_activations[l] = self.J_dot(self.d_activations[l][s],
                                              R_inputs[l][s])

            # copy output activations so we can reuse to compute error in
            # backwards pass
            R_outputs[s] = R_activations[-1]

        # R backward pass
        R_deltas = [np.zeros((self.inputs.shape[0], l), dtype=self.dtype)
                    for l in self.shape]
        for s in np.arange(sig_len - 1, -1, -1):
            # output layer
            if self.error_type == "mse":
                R_error = R_outputs[s]
            elif self.error_type == "ce":
                R_error = (R_outputs[s] *
                           np.nan_to_num(self.targets[:, s]) /
                           self.activations[-1][s] ** 2)

            if self.layer_types[-1].d_state is None:
                R_deltas[-1] = self.J_dot(self.d_activations[-1][s], R_error)
            else:
                R_deltas[-1] = (self.J_dot(self.d_activations[-1][s],
                                           R_error) +
                                self.layer_types[-1].d_state * R_deltas[-1])


            for l in np.arange(self.n_layers - 2, -1, -1):
                # feedforward gradient
                R_error = np.zeros_like(self.activations[l][s])
                for post in self.conns[l]:
                    W, _ = self.get_weights(self.W, (l, post))
                    R_error += np.dot(R_deltas[post], W.T)

                    offset, W_end, b_end = self.offsets[(l, post)]
                    Gv[offset:W_end] += (
                        self.outer_sum(self.activations[l][s]
                                       if self.GPU_activations is None
                                       else [l, s],
                                       R_deltas[post]))
                    Gv[W_end:b_end] += np.sum(R_deltas[post], axis=0)

                if l > 0:
                    # recurrent gradient
                    R_error += np.dot(R_deltas[l], W_recs[l][0].T)

                    if self.layer_types[l].d_state is None:
                        R_deltas[l] = self.J_dot(self.d_activations[l][s],
                                                 R_error)
                    else:
                        R_deltas[l] = (self.J_dot(self.d_activations[l][s],
                                                   R_error) +
                                       self.layer_types[l].d_state *
                                       R_deltas[l])


                    # apply structural damping
                    R_deltas[l] += self.J_dot(self.d_activations[l][s],
                                              damping * self.struc_damping *
                                              R_inputs[l][s])

                    offset, W_end, b_end = self.rec_offsets[l]
                    if s > 0:
                        Gv[offset:W_end] += (
                            self.outer_sum(self.activations[l][s - 1]
                                           if self.GPU_activations is None
                                           else [l, s - 1],
                                           R_deltas[l]))
                    else:
                        Gv[W_end:b_end] = np.sum(R_deltas[l], axis=0)

        Gv /= self.inputs.shape[0]

        Gv += damping * v  # Tikhonov damping

        return Gv
