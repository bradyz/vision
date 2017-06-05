from builtins import range
import numpy as np


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    s = conv_param['stride']
    p = conv_param['pad']

    N, _, H, W = x.shape
    F, _, HH, WW = w.shape
    H_ = 1 + (H + 2 * p - HH) // s
    W_ = 1 + (W + 2 * p - WW) // s

    pad = np.pad(x, ((0,0), (0,0), (p,p), (p,p)), 'constant')
    y = np.zeros((N, F, H_, W_))

    for n in range(N):
        for k in range(F):
            for i in range(H_):
                for j in range(W_):
                    y[n,k,i,j] = np.sum(np.multiply(pad[n,:,s*i:s*i+HH,s*j:s*j+WW], w[k]))
                    y[n,k,i,j] += b[k]

    return y, (x, w, b, conv_param)
