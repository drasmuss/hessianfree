"""Implementation of Hessian-free optimization for recurrent networks.

Author: Daniel Rasmussen (drasmussen@princeton.edu)

Based on
Martens, J., & Sutskever, I. (2011). Learning recurrent neural networks with
hessian-free optimization. Proceedings of the 28th International Conference on
Machine Learning.
"""

import numpy as np

from hessianbackprop import HessianBackprop


class HessianRNN(HessianBackprop):
    def __init__(self, struc_damping=0.0, **kwargs):
        # pretend that there are two hidden layers so that we get the
        # recurrent weight matrix
        layers = kwargs["layers"]
        assert len(layers) == 3
        kwargs["layers"] = [layers[0], layers[1], layers[1], layers[2]]

        self.struc_damping = struc_damping

        super(HessianRNN, self).__init__(**kwargs)

        # cut out the extra activation function if it was generated in super
        self.act = [self.act[0], self.act[-2], self.act[-1]]
        self.deriv = [self.deriv[0], self.deriv[-2], self.deriv[-1]]

    def forward(self, input, params):
        """Compute activations for given input sequence."""

        # input shape = [batch_size, seq_len, input_dim]
        # activations shape = [n_layers, seq_len, batch_size, layer_size]
        # note: seq_len and batch_size are swapped; this may be a bit confusing
        # but it makes the indexing a lot nicer in the rest of the code

        if len(input.shape) < 3:
            # then we've just been given a single sample (rather than batch)
            input = input[None, :, :]

        activations = [np.zeros((input.shape[1],
                                 input.shape[0],
                                 self.layers[i]),
                                dtype=(np.float32
                                       if not self.debug else
                                       np.float64))
                       for i in [0, 1, 3]]
        W_in, b_in = self.get_layer(params, 0)
        W_rec, b_rec = self.get_layer(params, 1)
        W_out, b_out = self.get_layer(params, 2)

        for s in range(input.shape[1]):
            # input activations
            activations[0][s] = self.act[0](input[:, s])

            # recurrent activations
            ff_input = np.dot(activations[0][s], W_in) + b_in

            if s > 0:
                rec_input = np.dot(activations[1][s - 1], W_rec)
            else:
                rec_input = b_rec
            activations[1][s] = self.act[1](ff_input + rec_input)

            # output activations
            activations[2][s] = self.act[2](np.dot(activations[1][s], W_out) +
                                             b_out)

        return activations

    def error(self, W=None, inputs=None, targets=None):
        """Compute RMS error."""

        W = self.W if W is None else W
        inputs = self.inputs if inputs is None else inputs
        targets = self.targets if targets is None else targets

        return super(HessianRNN, self).error(W, inputs,
                                             np.swapaxes(targets, 0, 1))

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
        rec_delta = np.zeros((inputs.shape[0], self.layers[1]),
                             dtype=np.float32)

        # backpropagate error
        for s in range(inputs.shape[1] - 1, -1, -1):
            # output layer
            if isinstance(activations[2], np.ndarray):
                error = activations[2][s] - targets[:, s]
            else:
                # then it's a GPU array, so use the non-GPU version (we don't
                # want to do this on the GPU)
                error = self.activations[2][s] - targets[:, s]

            delta = d_activations[2][s] * error

            offset, W_end, b_end = self.get_offsets(2)
            grad[offset:W_end] += self.outer_sum(activations[1][s], delta)
            grad[W_end:b_end] += np.sum(delta, axis=0)

            # recurrent layer
            error = (np.dot(delta, self.get_layer(W, 2)[0].T) +
                     np.dot(rec_delta, self.get_layer(W, 1)[0].T))
            rec_delta = d_activations[1][s] * error

            offset, W_end, b_end = self.get_offsets(1)
            if s > 0:
                # if s == 0 then the previous recurrent activations are zero,
                # so the gradient is zero
                grad[offset:W_end] += self.outer_sum(activations[1][s - 1],
                                                     rec_delta)

            # input layer
            offset, W_end, b_end = self.get_offsets(0)
            grad[offset:W_end] += self.outer_sum(activations[0][s], rec_delta)
            grad[W_end:b_end] += np.sum(rec_delta, axis=0)

        # this is the bias input used to initialize the hidden layer
        # activations on the first timestep
        _, W_end, b_end = self.get_offsets(1)
        grad[W_end:b_end] = np.sum(rec_delta, axis=0)

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
            assert acts[2].shape[2] == 1

            base = np.concatenate((acts[2].squeeze(axis=(1, 2)),
                                   acts[1][-1].squeeze(axis=0)))

            J = np.zeros((sig_len + self.layers[1], N))
            for i in range(N):
                inc_i = np.zeros(N)
                inc_i[i] = eps

                acts = self.forward(inputs[b], self.W + inc_i)
                J[:, i] = (np.concatenate((acts[2].squeeze(axis=(1, 2)),
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

        Ws = [self.get_layer(self.W, i)[0] for i in range(3)]
        vW_in, vb_in = self.get_layer(v, 0)
        vW_rec, vb_rec = self.get_layer(v, 1)
        vW_out, vb_out = self.get_layer(v, 2)

        # R forward pass
        R_inputs = np.zeros(self.activations[1].shape, dtype=np.float32)
        R_hidden = np.zeros((self.inputs.shape[0], self.layers[1]),
                            dtype=np.float32)
        R_outputs = np.zeros(self.activations[2].shape, dtype=np.float32)
        for s in np.arange(sig_len):
            if s == 0:
                R_inputs[s] = (np.dot(self.activations[0][s], vW_in) + vb_in +
                               vb_rec)
            else:
                R_inputs[s] = (np.dot(self.activations[0][s], vW_in) + vb_in +
                               np.dot(self.activations[1][s - 1], vW_rec) +
                               np.dot(R_hidden, Ws[1]))

            R_hidden = self.d_activations[1][s] * R_inputs[s]

            R_outputs[s] = (self.d_activations[2][s] *
                            (np.dot(self.activations[1][s], vW_out) + vb_out +
                             np.dot(R_hidden, Ws[2])))

        # R backward pass
        R_rec_delta = np.zeros((self.inputs.shape[0], self.layers[1]),
                               dtype=np.float32)
        for s in np.arange(sig_len - 1, -1, -1):
            # output layer
            R_delta = self.d_activations[2][s] * R_outputs[s]
            # note: this is different than the martens pseudocode, but that's
            # because he uses linear output units

            offset, W_end, b_end = self.get_offsets(2)
            Gv[offset:W_end] += self.outer_sum(self.activations[1][s],
                                               R_delta)
            Gv[W_end:b_end] += np.sum(R_delta, axis=0)

            # hidden layer
            offset, W_end, b_end = self.get_offsets(1)
            Gv[offset:W_end] += self.outer_sum(self.activations[1][s],
                                               R_rec_delta)

            R_rec_delta = (np.dot(R_rec_delta, Ws[1].T) +
                           np.dot(R_delta, Ws[2].T))
            R_rec_delta *= self.d_activations[1][s]

            # apply structural damping
            R_rec_delta += (damping * self.struc_damping *
                            self.d_activations[1][s] * R_inputs[s])

            # input layer
            offset, W_end, b_end = self.get_offsets(0)
            Gv[offset:W_end] += self.outer_sum(self.activations[0][s],
                                               R_rec_delta)
            Gv[W_end:b_end] += np.sum(R_rec_delta, axis=0)

        _, W_end, b_end = self.get_offsets(1)
        Gv[W_end:b_end] = np.sum(R_rec_delta, axis=0)

        Gv /= self.inputs.shape[0]

        Gv += damping * v  # Tikhonov damping

        return Gv
