#include <iostream>
#include "library.h"

///////////////////////////////////// declaration

template<typename T>
void im2col(const T* data_im, const int channels,
            const int height, const int width,
            const int kernel_k, T* data_col);

template<typename T>
void matrix_mul(T* A, int a_rows, int a_cols,
                T* B, int b_rows, int b_cols,
                T* C, int c_rows, int c_cols);

///////////////////////////////////// instantiation
template
void conv2d_im2col(tensor_4d_input<float>& ti, tensor_4d_kernel<float>& tk,
                   tensor_4d_output<float>& to);

template
void im2col(const float* data_im, const int channels,
            const int height, const int width,
            const int kernel_k, float* data_col);

template
void matrix_mul(float* A, int a_rows, int a_cols,
                float* B, int b_rows, int b_cols,
                float* C, int c_rows, int c_cols);

////////////////////////////////////////////////////////////////////////////////////
/**
 * Tensor convolution using im2col and matrix multiplication
 * https://towardsdatascience.com/how-are-convolutions-actually-performed-under-the-hood-226523ce7fbf
 * @param input tensor 4d
 * @param kernel tensor 4d
 * @param output tensor 4d
 */
template<typename T>
void conv2d_im2col(tensor_4d_input<T>& ti,
                   tensor_4d_kernel<T>& tk,
                   tensor_4d_output<T>& to)
{
    // im2col buffer
    T* ti_im2col = new T[(ti.H_ * ti.W_) * (ti.Ci_ * tk.K_ * tk.K_)];
    // main loop
    for (int n = 0; n < ti.N_; n++)
    {
        int in_offs = n * ti.Ci_ * ti.H_ * ti.W_;
        im2col(ti.data_ + in_offs, ti.Ci_, ti.H_, ti.W_, tk.K_, ti_im2col);
        // matrix mul
        int out_offs = n * to.Co_ * ti.H_ * ti.W_;
        matrix_mul(tk.data_, tk.Co_, tk.Ci_ * tk.K_ * tk.K_,
                   ti_im2col, tk.Ci_ * tk.K_ * tk.K_, ti.H_ * ti.W_,
                   to.data_ + out_offs, tk.Co_, ti.H_ * ti.W_);
    }
    delete [] ti_im2col;
}


static inline bool is_a_ge_zero_and_a_lt_b(int a, int b)
{
    return (unsigned int)a < (unsigned int)(b);
}

/**
 * im2col transformation - unroll input regions for filtering
 * https://www.programmersought.com/article/7026378651/
 * Inspired by Berkeley Vision's Caffe
 * https://github.com/BVLC/caffe/blob/master/LICENSE
 * @tparam T - element type
 * @param data_im
 * @param channels
 * @param height
 * @param width
 * @param kernel_k
 * @param data_col
 */
template<typename T>
void im2col(const T* data_im, const int channels,
            const int height, const int width,
            const int kernel_k, T* data_col)
{
    int pad = (kernel_k - 1) / 2;
    assert(2 * pad + 1 == kernel_k);
    const int output_h = height;
    const int output_w = width;
    const int channel_size = height * width;
    for(int channel = channels; channel--; data_im += channel_size)
    {
        for(int kernel_row = 0; kernel_row < kernel_k; kernel_row++)
        {
            for(int kernel_col = 0; kernel_col < kernel_k; kernel_col++)
            {
                int input_row = -pad + kernel_row;
                for(int output_rows = output_h; output_rows; output_rows--)
                {
                    if(!is_a_ge_zero_and_a_lt_b(input_row, height))
                    {
                        for(int output_cols = output_w; output_cols; output_cols--)
                            *(data_col++) = 0; // compiler will cast to T type automatically
                    }
                    else
                    {
                        int input_col = -pad + kernel_col;
                        for(int output_col = output_w; output_col; output_col--)
                        {
                            if(is_a_ge_zero_and_a_lt_b(input_col, width))
                                *(data_col++) = data_im[input_row * width + input_col];
                            else
                                *(data_col++) = 0; // compiler will cast to T type automatically
                            input_col += 1;
                        }
                    }
                    input_row += 1;
                }
            }
        }
    }
}

/**
 * Naive implementation of matrix multiplication
 */
template<typename T>
void matrix_mul(T* A, int a_rows, int a_cols,
                T* B, int b_rows, int b_cols,
                T* C, int c_rows, int c_cols)
{
    assert(a_cols == b_rows);
    assert(c_rows == a_rows);
    assert(c_cols == b_cols);
    for (int r = 0; r < a_rows; r++)
    {
        for (int c = 0; c < b_cols; c++)
        {
            T sum = 0;  // compiler will cast to T type automatically
            for(int ac = 0; ac < a_cols; ac++)
                sum += A[ac * a_rows + r] * B[c * a_cols + ac];
            C[c * a_rows + r] = sum;
        }
    }
}
