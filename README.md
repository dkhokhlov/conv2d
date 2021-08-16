# 2D Convolution via Matrix Multiplication

Convolutional Neural Networks (CNNs) are typically used for analyzing images. Typically, CNNs for still images,
e.g. ResNet, consist of repeated applications of 2-dimensional multi-channel convolutions. The convolution
operations can be implemented using direct algorithms, transform-based algorithms (e.g., discrete Fourier
transform, Winograd’s transform), or tensor-multiplication-based algorithms, which are easily amenable to
hardware acceleration. In terms of the tensor-multiplication-based algorithms, `im2col` and `kn2row` are some of the most
well-studied algorithms for turning a convolution operation into a simple multiplication between two tensors.
The transformation from convolution to a tensor multiplication is achieved via data manipulation and replication.

## Data formats
### Input
- A 4-dimensional kernel with shape (Cout x Cin x k x k), where:
  - Cout is the number of output channels
  - Cin is the number of input channels
  - k is the height and width of the kernel
- A 4-dimensional input tensor with shape (N x Cin x H x W) where:
  - N is the batch size
  - Cin is the number of input channels
  - H is the spatial height of the input (image) tensor
  - W is the spatial width of the input (image) tensor
### Output
- A 4-dimensional output tensor with shape (N x Cout x H x W) where:
  - N is the batch size
  - Cout is the number of output channels
  - H is the spatial height of the input (image) tensor
  - W is the spatial width of the input (image) tensor

## Design notes
- modules:
  - lightmatter_conv.so - native module using [NumPy C-API](https://numpy.org/doc/stable/reference/c-api/index.html) (not automaticaly generated), deployed into site packages.
  - ./conv.py - python module, provides function conv2d, that is based on im2conv
  - ./test.py - unit tests
- optinally allows to add Gaussian noise to output

## Running
Requires:
- CMmake >=3.19.6
- numpy
```
./mk && ./mk.deploy && ./mk.test
```
## ToDo
- add more tests with result verification
- replace naive matrix multiplication by SGEMM

