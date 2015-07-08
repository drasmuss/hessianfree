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
        # TODO: allow user to specify which layers are recurrent, instead
        # of assuming all except first/last

        self.struc_damping = struc_damping

        # add on recurrent weights
        self.W = np.concatenate(
            (self.W, self.init_weights([(l + 1, l)
                                        for l in self.layers[1:-1]])))

    def compute_offsets(self):
        """Precompute offsets for layers in the overall parameter vector."""

        super(HessianRNN, self).compute_offsets()

        self.rec_offsets = {}

        for l in range(1, self.n_layers - 1):
            offset = (len(self.W) +  # note: gets called before rec_W added
                      np.sum([(self.layers[i] + 1) * self.layers[i]
                              for i in range(1, l)]))
            self.rec_offsets[l] = (
                offset,
                offset + self.layers[l] * self.layers[l],
                offset + (self.layers[l] + 1) * self.layers[l])

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

                return (W.reshape((self.layers[layer],
                                   self.layers[layer])),
                        b)
            else:
                return params[offset:b_end].reshape((self.layers[layer] + 1,
                                                     self.layers[layer]))

    def forward(self, input, params):
        """Compute activations for given input sequence and parameters."""

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

        W_recs = [self.get_weights(params, i, recurrent=True)
                  for i in np.arange(self.n_layers)]
        for s in range(input.shape[1]):
            # input activations
            activations[0][s] = self.act[0](input[:, s])

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
        deltas = [np.zeros((inputs.shape[0], l), dtype=np.float32)
                  for l in self.layers]
        W_rec = [self.get_weights(W, l, recurrent=True)
                 for l in np.arange(self.n_layers - 1)]

        # backpropagate error
        for s in range(inputs.shape[1] - 1, -1, -1):
            if isinstance(activations[-1], np.ndarray):
                error = activations[-1][s] - targets[:, s]
            else:
                # then it's a GPU array, so use the non-GPU version (we don't
                # want to do this on the GPU)
                error = self.activations[-1][s] - targets[:, s]

            error = np.nan_to_num(error)  # zero error where target==nan

            deltas[-1][...] = d_activations[-1][s] * error
            for l in range(self.n_layers - 2, -1, -1):
                # gradient for output weights
                error = np.zeros_like(deltas[l])
                for post in self.conns[l]:
                    c_error = np.dot(deltas[post],
                                     self.get_weights(W, (l, post))[0].T)
                    error += c_error
                    offset, W_end, b_end = self.offsets[(l, post)]
                    grad[offset:W_end] += self.outer_sum(activations[l][s],
                                                         deltas[post])
                    grad[W_end:b_end] += np.sum(deltas[post], axis=0)

                if l > 0:
                    # gradient for recurrent weights
                    error += np.dot(deltas[l], W_rec[l][0].T)
                    deltas[l][...] = d_activations[l][s] * error

                    offset, W_end, b_end = self.rec_offsets[l]
                    if s > 0:
                        grad[offset:W_end] += (
                            self.outer_sum(activations[l][s - 1], deltas[l]))
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

        if output is None:
            Gv = np.zeros(self.W.size, dtype=np.float32)
        else:
            Gv = output
            Gv[:] = 0

        sig_len = self.inputs.shape[1]

        # R forward pass
        R_inputs = [np.zeros(self.activations[i].shape, dtype=np.float32)
                    for i in np.arange(self.n_layers)]
        R_activations = [np.zeros((self.inputs.shape[0], self.layers[i]),
                                  dtype=np.float32)
                         for i in np.arange(self.n_layers)]
        R_outputs = np.zeros(self.activations[-1].shape, dtype=np.float32)
        v_recs = [self.get_weights(v, l, recurrent=True)
                  for l in np.arange(self.n_layers - 1)]
        W_recs = [self.get_weights(self.W, l, recurrent=True)
                  for l in np.arange(self.n_layers - 1)]
        for s in np.arange(sig_len):
            for l in np.arange(1, self.n_layers):
                for pre in self.back_conns[l]:
                    vw, vb = self.get_weights(v, (pre, l))
                    Ww, _ = self.get_weights(self.W, (pre, l))
                    R_inputs[l][s] += np.dot(self.activations[pre][s],
                                             vw) + vb
                    R_inputs[l][s] += np.dot(R_activations[pre], Ww)

                if l < self.n_layers - 1:
                    # add recurrent input
                    if s == 0:
                        R_inputs[l][s] += v_recs[l][1]
                    else:
                        R_inputs[l][s] += (
                            np.dot(self.activations[l][s - 1],
                                   v_recs[l][0]) +
                            np.dot(R_activations[l], W_recs[l][0]))

                R_activations[l][...] = (self.d_activations[l][s] *
                                         R_inputs[l][s])

            # copy output activations so we can reuse to compute error in
            # backwards pass
            R_outputs[s] = R_activations[-1]

        # R backward pass
        R_deltas = [np.zeros((self.inputs.shape[0], self.layers[l]),
                             dtype=np.float32)
                    for l in np.arange(self.n_layers)]
        for s in np.arange(sig_len - 1, -1, -1):
            # output layer
            R_deltas[-1][...] = self.d_activations[-1][s] * R_outputs[s]

            for l in np.arange(self.n_layers - 2, -1, -1):
                # feedforward gradient
                R_error = np.zeros_like(self.activations[l][s])
                for post in self.conns[l]:
                    W, _ = self.get_weights(self.W, (l, post))
                    R_error += np.dot(R_deltas[post], W.T)

                    offset, W_end, b_end = self.offsets[(l, post)]
                    if self.use_GPU:
                        Gv[offset:W_end] += (
                            self.outer_sum(self.GPU_activations[l][s],
                                           R_deltas[post]))
                    else:
                        Gv[offset:W_end] += (
                            self.outer_sum(self.activations[l][s],
                                           R_deltas[post]))
                    Gv[W_end:b_end] += np.sum(R_deltas[post], axis=0)

                if l > 0:
                    # recurrent gradient
                    R_error += np.dot(R_deltas[l], W_recs[l][0].T)
                    R_deltas[l][...] = self.d_activations[l][s] * R_error

                    # apply structural damping
                    R_deltas[l] += (damping * self.struc_damping *
                                    self.d_activations[l][s] *
                                    R_inputs[l][s])

                    offset, W_end, b_end = self.rec_offsets[l]
                    if s > 0:
                        if self.use_GPU:
                            Gv[offset:W_end] += (
                                self.outer_sum(self.GPU_activations[l][s - 1],
                                               R_deltas[l]))
                        else:
                            Gv[offset:W_end] += (
                                self.outer_sum(self.activations[l][s - 1],
                                               R_deltas[l]))
                    else:
                        Gv[W_end:b_end] = np.sum(R_deltas[l], axis=0)

        Gv /= self.inputs.shape[0]

        Gv += damping * v  # Tikhonov damping

        return Gv
