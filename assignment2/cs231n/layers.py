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
                    y[n,k,i,j] = np.sum(np.multiply(pad[n,:,s*i:s*i+HH,s*j:s*j+WW],
                                                    w[k]))
                    y[n,k,i,j] += b[k]

    return y, (x, w, b, conv_param)


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None

    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    s = pool_param['stride']
    h = pool_param['pool_height']
    w = pool_param['pool_width']

    N, C, H, W = x.shape
    H_ = H // s
    W_ = W // s

    y = np.zeros((N, C, H_, W_))

    for n in range(N):
        for k in range(C):
            for i in range(H_):
                for j in range(W_):
                    y[n,k,i,j] = np.max(x[n,k,i*s:i*s+h,j*s:j*s+w])

    return y, (x, pool_param)


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    x, pool_param = cache
    s = pool_param['stride']
    h = pool_param['pool_height']
    w = pool_param['pool_width']

    N, C, H, W = x.shape
    H_ = H // s
    W_ = W // s

    dx = np.zeros((x.shape))

    for n in range(N):
        for k in range(C):
            for i in range(H_):
                for j in range(W_):
                    dx[n,k,i*s:i*s+h,j*s:j*s+w] = dout[n,k,i,j]

                    pool = np.max(x[n,k,i*s:i*s+h,j*s:j*s+w])
                    chunk = x[n,k,i*s:i*s+h,j*s:j*s+w]

                    dx[n,k,i*s:i*s+h,j*s:j*s+w] *= (pool == chunk)

    return dx
