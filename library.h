#pragma once
#include <numpy/arrayobject.h>

template<typename T>
struct tensor_4d {
public:
    tensor_4d(T* data, const npy_intp* strides, const npy_intp* dims) :
            data_(data),
            strides_(strides),
            dims_(dims)
    {
    }
public:
    T* data_; // frame
    const npy_intp* strides_; // stride per each dimension
    const npy_intp* dims_;  // array of dimensions
};

template<typename T>
struct tensor_4d_input : public tensor_4d<T> {
public:
    tensor_4d_input(T* data, const npy_intp* strides, const npy_intp* dims) : tensor_4d<T>(data, strides, dims),
        N_(dims[0]), Ci_(dims[1]), H_(dims[2]), W_(dims[3])
    {
    }
public:
    const npy_intp& N_; // batch
    const npy_intp& Ci_; // input channels
    const npy_intp& H_; // rows
    const npy_intp& W_; // cols
};

template<typename T>
struct tensor_4d_kernel : public tensor_4d<T> {
public:
    tensor_4d_kernel(T* data, const npy_intp* strides, const npy_intp* dims) : tensor_4d<T>(data, strides, dims),
    Co_(dims[0]), Ci_(dims[1]), K_(dims[2])
    {
        assert(dims[2] == dims[3]);
    }
public:
    const npy_intp& Co_; // output channels
    const npy_intp& Ci_; // input channels
    const npy_intp& K_; // kernel size
};

template<typename T>
struct tensor_4d_output : public tensor_4d<T> {
public:
    tensor_4d_output(T* data, const npy_intp* strides, const npy_intp* dims) : tensor_4d<T>(data, strides, dims),
    N_(dims[0]), Co_(dims[1]), H_(dims[2]), W_(dims[3])
    {
    }
public:
    const npy_intp& N_; // batch size
    const npy_intp& Co_; // output channels
    const npy_intp& H_; // rows
    const npy_intp& W_; // cols
};

template<typename T>
void conv2d_im2col(tensor_4d_input<T>& ti, tensor_4d_kernel<T>& tk,
                   tensor_4d_output<T>& to);
