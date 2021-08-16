import libconv2d  # native module
import numpy as np


def conv2d(image: np.ndarray, kernel: np.ndarray, output: np.ndarray,
           random_bias: tuple = None, conv_type='im2col'):
    """
    2D convolution of 4D tensors.
    Args:
        image: 4D tensor with shape [N, Ci, H, W], N - batch, Ci - input channels
        kernel: 4D tensor with shape [Co, Ci, K, K], odd K size kernel is expected. Co - output channels
        output: 4D tensor with shape [N, Co, H, W], must be pre-allocated with proper size.
        random_bias: (mean, sigma) - add random bias sampled from normal distribution. Default: None, not added.
        conv_type: type of convolution algorithm implementation. Default: 'im2col'.
    Returns:
    """
    if conv_type != 'im2col':
        raise Exception(f"conv_type='{conv_type}' - not implemented")
    # validate params
    assert len(image.shape) == 4
    assert len(kernel.shape) == 4
    assert len(output.shape) == 4
    iN, iCi, iH, iW = image.shape
    kCo, kCi, kW, kH = kernel.shape
    oN, oCo, oH, oW = output.shape
    assert (iH, iW) == (oH, oW)
    assert iN == oN
    assert iCi == kCi
    assert kCo == oCo
    assert kH == kW
    assert kH <= iH and kH <= iW
    pad = (kH - 1) / 2
    assert 2 * pad + 1 == kH
    # call C++ implementation
    libconv2d.conv2d_im2col(image, kernel, output)
    if random_bias:
        output += np.random.normal(random_bias[0], random_bias[1], output.shape)
