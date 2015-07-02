"""Implementation of Hessian-free optimization for recurrent networks.

Author: Daniel Rasmussen (drasmussen@princeton.edu)

Based on
Martens, J., & Sutskever, I. (2011). Learning recurrent neural networks with
hessian-free optimization. Proceedings of the 28th International Conference on
Machine Learning.
"""

import numpy as np

from hessianff import HessianFF


class HessianRNN(HessianFF):
    def __init__(self, struc_damping=0.0, **kwargs):
        super(HessianRNN, self).__init__(**kwargs)

        self.struc_damping = struc_damping

        self.W = np.concatenate(
            (self.W,
             self.init_weights([(l + 1, l) for l in self.layers[1:-1]])))

    def get_offsets(self, layer, recurrent=False):
        if not recurrent:
            return super(HessianRNN, self).get_offsets(layer)
        else:
            # no recurrent weights for first/last layer
            assert layer != 0 and layer != len(self.layers) - 1

            offset = (np.sum(self.n_params) +
                      np.sum([(self.layers[i] + 1) * self.layers[i]
                              for i in range(1, layer)]))
            return (offset,
                    offset + self.layers[layer] * self.layers[layer],
                    offset + (self.layers[layer] + 1) * self.layers[layer])

    def get_weights(self, params, layer, separate=True, recurrent=False):
        if not recurrent:
            return super(HessianRNN, self).get_weights(params, layer, separate)
        else:
            offset, W_end, b_end = self.get_offsets(layer, recurrent)
            if separate:
                W = params[offset:W_end]
                b = params[W_end:b_end]

                return (W.reshape((self.layers[layer],
                                   self.layers[layer])),
                        b)
            else:
                return params[offset:b_end].reshape((self.layers[layer] + 1,
                                                     self.layers[layer]))

    def forward(self, input, params):
        """Compute activations for given input sequence."""

        # input shape = [batch_size, seq_len, input_dim]
        # activations shape = [n_layers, seq_len, batch_size, layer_size]
        # note: seq_len and batch_size are swapped; this may be a bit confusing
        # but it makes the indexing a lot nicer in the rest of the code

        if input.ndim < 3:
            # then we've just been given a single sample (rather than batch)
            input = input[None, :, :]

        activations = [np.zeros((input.shape[1],
                                 input.shape[0],
                                 l),
                                dtype=(np.float32
                                       if not self.debug else
                                       np.float64))
                       for l in self.layers]

        for s in range(input.shape[1]):
            # input activations
            activations[0][s] = self.act[0](input[:, s])

            for i in range(self.n_layers - 1):
                W, b = self.get_weights(params, i)
                ff_input = np.dot(activations[i][s], W) + b

                if i == self.n_layers - 2:
                    # no recurrent connections for last layer
                    rec_input = 0
                else:
                    W_rec, b_rec = self.get_weights(params, i + 1,
                                                    recurrent=True)
                    if s > 0:
                        rec_input = np.dot(activations[i + 1][s - 1], W_rec)
                    else:
                        # bias input on first timestep
                        rec_input = b_rec

                activations[i + 1][s] = self.act[i + 1](ff_input + rec_input)

        return activations

    def error(self, W=None, inputs=None, targets=None):
        """Compute RMS error."""

        W = self.W if W is None else W
        inputs = self.inputs if inputs is None else inputs
        targets = self.targets if targets is None else targets
        targets = np.swapaxes(targets, 0, 1)

        return super(HessianRNN, self).error(W, inputs, targets)

    def calc_grad(self, W=None, inputs=None, targets=None):
        """Compute parameter gradient."""

        # TODO: check this function for possible optimizations

        W = self.W if W is None else W
        inputs = self.inputs if inputs is None else inputs
        targets = self.targets if targets is None else targets

        if W is self.W and inputs is self.inputs:
            # use cached activations
            activations = (self.activations
                           if not self.use_GPU else
                           self.GPU_activations)
            d_activations = self.d_activations
        else:
            # compute activations
            activations = self.forward(inputs, W)
            d_activations = [self.deriv[i](a)
                             for i, a in enumerate(activations)]

        grad = np.zeros(W.size, dtype=np.float32)
        delta = [np.zeros((inputs.shape[0], l), dtype=np.float32)
                 for l in self.layers]

        # backpropagate error
        for s in range(inputs.shape[1] - 1, -1, -1):
            if isinstance(activations[-1], np.ndarray):
                error = activations[-1][s] - targets[:, s]
            else:
                # then it's a GPU array, so use the non-GPU version (we don't
                # want to do this on the GPU)
                error = self.activations[-1][s] - targets[:, s]

            error = np.nan_to_num(error)  # zero error where target==nan

            delta[-1] = d_activations[-1][s] * error
            for l in range(self.n_layers - 2, -1, -1):
                # gradient for output weights
                offset, W_end, b_end = self.get_offsets(l)

                grad[offset:W_end] += self.outer_sum(activations[l][s],
                                                     delta[l + 1])
                grad[W_end:b_end] += np.sum(delta[l + 1], axis=0)

                if l > 0:
                    # gradient for recurrent weights
                    W_ff, _ = self.get_weights(W, l)
                    W_rec, _ = self.get_weights(W, l, recurrent=True)
                    error = (np.dot(delta[l + 1], W_ff.T) +
                             np.dot(delta[l], W_rec.T))
                    delta[l] = d_activations[l][s] * error

                    offset, W_end, b_end = self.get_offsets(l, recurrent=True)
                    if s > 0:
                        grad[offset:W_end] += (
                            self.outer_sum(activations[l][s - 1], delta[l]))
                    else:
                        # put remaining gradient into initial bias
                        grad[W_end:b_end] = np.sum(delta[l], axis=0)

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

            base = np.concatenate((acts[-1].squeeze(axis=(1, 2)),
                                   acts[1][-1].squeeze(axis=0)))

            J = np.zeros((sig_len + self.layers[1], N))
            for i in range(N):
                inc_i = np.zeros(N)
                inc_i[i] = eps

                acts = self.forward(inputs[b], self.W + inc_i)
                J[:, i] = (np.concatenate((acts[-1].squeeze(axis=(1, 2)),
                                           acts[1][-1].squeeze(axis=0))) -
                           base) / eps

            L = np.diag([1] * sig_len + [self.struc_damping] * self.layers[1])

            g += np.dot(np.dot(J.T, np.dot(L, J)), v)

        g /= inputs.shape[0]

        g += damping * v

        try:
            assert np.allclose(g, calc_G, atol=1e-6)
        except AssertionError:
            print g
            print calc_G
            print g - calc_G
            print g / calc_G
            raise

    def G(self, v, damping=0, output=None):
        """Compute Gauss-Newton matrix-vector product."""

        # TODO: check this function for possible optimizations

        if output is None:
            Gv = np.zeros(self.W.size, dtype=np.float32)
        else:
            Gv = output
            Gv[:] = 0

        sig_len = self.inputs.shape[1]

        # R forward pass
        R_inputs = [np.zeros(self.activations[i].shape, dtype=np.float32)
                    for i in np.arange(self.n_layers)]
        R_acts = [np.zeros((self.inputs.shape[0], self.layers[i]),
                           dtype=np.float32)
                  for i in np.arange(self.n_layers)]
        R_outputs = np.zeros(self.activations[-1].shape, dtype=np.float32)
        vs = [self.get_weights(v, l)
              for l in np.arange(self.n_layers - 1)]
        Ws = [self.get_weights(self.W, l)
              for l in np.arange(self.n_layers - 1)]
        for s in np.arange(sig_len):
            for l in np.arange(self.n_layers - 1):
                R_inputs[l + 1][s] = np.dot(self.activations[l][s],
                                            vs[l][0]) + vs[l][1]
                R_inputs[l + 1][s] += np.dot(R_acts[l], Ws[l][0])

                if l < self.n_layers - 2:
                    v_rec, v_rec_b = self.get_weights(v, l + 1, recurrent=True)
                    W_rec, _ = self.get_weights(self.W, l + 1, recurrent=True)

                    # add recurrent input
                    if s == 0:
                        R_inputs[l + 1][s] += v_rec_b
                    else:
                        R_inputs[l + 1][s] += (
                            np.dot(self.activations[l + 1][s - 1], v_rec) +
                            np.dot(R_acts[l + 1], W_rec))

                R_acts[l + 1] = (self.d_activations[l + 1][s] *
                                 R_inputs[l + 1][s])
            R_outputs[s, ...] = R_acts[-1]  # copy so we can reuse in back pass

        # R backward pass
        R_delta = [np.zeros((self.inputs.shape[0], self.layers[l]),
                            dtype=np.float32)
                   for l in np.arange(self.n_layers)]
        for s in np.arange(sig_len - 1, -1, -1):
            # output layer
            R_delta[-1] = self.d_activations[-1][s] * R_outputs[s]

            for l in np.arange(self.n_layers - 2, -1, -1):
                # feedforward gradient
                offset, W_end, b_end = self.get_offsets(l)
                if self.use_GPU:
                    Gv[offset:W_end] += (
                        self.outer_sum(self.GPU_activations[l][s],
                                       R_delta[l + 1]))
                else:
                    Gv[offset:W_end] += self.outer_sum(self.activations[l][s],
                                                       R_delta[l + 1])
                Gv[W_end:b_end] += np.sum(R_delta[l + 1], axis=0)

                if l > 0:
                    # recurrent gradient
                    W, _ = self.get_weights(self.W, l)
                    W_rec, _ = self.get_weights(self.W, l, recurrent=True)
                    R_delta[l] = (np.dot(R_delta[l], W_rec.T) +
                                  np.dot(R_delta[l + 1], W.T))
                    R_delta[l] *= self.d_activations[l][s]

                    # apply structural damping
                    R_delta[l] += (damping * self.struc_damping *
                                   self.d_activations[l][s] *
                                   R_inputs[l][s])

                    offset, W_end, b_end = self.get_offsets(l, recurrent=True)
                    if s > 0:
                        if self.use_GPU:
                            Gv[offset:W_end] += (
                                self.outer_sum(self.GPU_activations[l][s - 1],
                                               R_delta[l]))
                        else:
                            Gv[offset:W_end] += (
                                self.outer_sum(self.activations[l][s - 1],
                                               R_delta[l]))
                    else:
                        Gv[W_end:b_end] = np.sum(R_delta[l], axis=0)

        Gv /= self.inputs.shape[0]

        Gv += damping * v  # Tikhonov damping

        return Gv
