import numpy

from chainer.functions.activation import hard_sigmoid
from chainer.functions.activation import tanh
from chainer.functions.math import linear_interpolate
from chainer import link
from chainer.links.connection import linear
from chainer import variable

"""
Copyright (c) 2015 Preferred Infrastructure, Inc.
Copyright (c) 2015 Preferred Networks, Inc.

Pls see LICENSE-chainer.txt in the 'docs' directory

This is a change of
<\python3\Lib\site-packages\chainer\links\connection\gru.py>

Change: to be similar theano's gru
       a) sigmoid->hard_sigmoid
       b) every one has bias-> only one for each
       C) h_next= (1-z) * h + z * h_ -> h_next=  z * h + (1-z)*h_
       ---
       d) remove statelessGRU
Date: 2018.6
"""

class GRUBase2(link.Chain):

    def __init__(self, in_size, out_size,
                 init=None,   inner_init=None,   bias_init=None,
                 init_r=None, inner_init_r=None, bias_init_r=None,
                 init_z=None, inner_init_z=None, bias_init_z=None):
        super(GRUBase2, self).__init__()
        with self.init_scope():
            self.W =   linear.Linear(in_size,  out_size, initialW=init,         initial_bias=bias_init  )
            self.U =   linear.Linear(out_size, out_size, initialW=inner_init,   nobias=True)    #initial_bias=bias_init)
            self.W_r = linear.Linear(in_size,  out_size, initialW=init_r,       initial_bias=bias_init_r)
            self.U_r = linear.Linear(out_size, out_size, initialW=inner_init_r, nobias=True)  #initial_bias=bias_init)
            self.W_z = linear.Linear(in_size,  out_size, initialW=init_z,       initial_bias=bias_init_z)
            self.U_z = linear.Linear(out_size, out_size, initialW=inner_init_z, nobias=True)   #initial_bias=bias_init)


class StatefulGRU2(GRUBase2):
    """Stateful Gated Recurrent Unit function (GRU).

    Stateful GRU function has six parameters :math:`W_r`, :math:`W_z`,
    :math:`W`, :math:`U_r`, :math:`U_z`, and :math:`U`.
    All these parameters are :math:`n \\times n` matrices,
    where :math:`n` is the dimension of hidden vectors.

    Given input vector :math:`x`, Stateful GRU returns the next
    hidden vector :math:`h'` defined as

    .. math::

       r &=& \\sigma(W_r x + U_r h), \\\\
       z &=& \\sigma(W_z x + U_z h), \\\\
       \\bar{h} &=& \\tanh(W x + U (r \\odot h)), \\\\
       h' &=& (1 - z) \\odot h + z \\odot \\bar{h},

    where :math:`h` is current hidden vector.

    As the name indicates, :class:`~chainer.links.StatefulGRU` is *stateful*,
    meaning that it also holds the next hidden vector `h'` as a state.
    For a *stateless* GRU, use :class:`~chainer.links.StatelessGRU`.

    Args:
        in_size(int): Dimension of input vector :math:`x`.
        out_size(int): Dimension of hidden vector :math:`h`.
        init: Initializer for GRU's input units (:math:`W`).
            It is a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
            If it is ``None``, the default initializer is used.
        inner_init: Initializer for the GRU's inner
            recurrent units (:math:`U`).
            It is a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
            If it is ``None``, the default initializer is used.
        bias_init: Bias initializer.
            It is a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
            If ``None``, the bias is set to zero.

    Attributes:
        h(~chainer.Variable): Hidden vector that indicates the state of
            :class:`~chainer.links.StatefulGRU`.

    .. seealso::
        * :class:`~chainer.links.StatelessGRU`
        * :class:`~chainer.links.GRU`: an alias of
          :class:`~chainer.links.StatefulGRU`

    """

    def __init__(self, in_size, out_size, 
                 init=None,   inner_init=None,   bias_init=0, 
                 init_r=None, inner_init_r=None, bias_init_r=0,
                 init_z=None, inner_init_z=None, bias_init_z=0):
        super(StatefulGRU2, self).__init__(
            in_size, out_size, init, inner_init, bias_init,init_r, inner_init_r, bias_init_r,init_z, inner_init_z, bias_init_z )
        self.state_size = out_size
        self.reset_state()

    def to_cpu(self):
        super(StatefulGRU2, self).to_cpu()
        if self.h is not None:
            self.h.to_cpu()

    def to_gpu(self, device=None):
        super(StatefulGRU2, self).to_gpu(device)
        if self.h is not None:
            self.h.to_gpu(device)

    def set_state(self, h):
        assert isinstance(h, variable.Variable)
        h_ = h
        if self.xp == numpy:
            h_.to_cpu()
        else:
            h_.to_gpu(self._device_id)
        self.h = h_

    def reset_state(self):
        self.h = None

    def __call__(self, x):
        z = self.W_z(x)
        h_bar = self.W(x)
        if self.h is not None:
            r = hard_sigmoid.hard_sigmoid(self.W_r(x) + self.U_r(self.h))
            z += self.U_z(self.h)
            h_bar += self.U(r * self.h)  # this may differs by version
        z = hard_sigmoid.hard_sigmoid(z)
        h_bar = tanh.tanh(h_bar)

        if self.h is not None:
            h_new = linear_interpolate.linear_interpolate(z, self.h, h_bar)  #(z, h_bar, self.h)
        else:
            h_new = ( 1- z) * h_bar
        self.h = h_new
        
        return self.h


