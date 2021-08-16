/*  Example of wrapping the cos function from math.h using the Numpy-C-API. */
#include <Python.h>
#include <numpy/arrayobject.h>
#include "library.h"

static PyObject* conv2d_im2col_wrapper(PyObject*, PyObject* args)
{
    PyObject *arg1=NULL, *arg2=NULL, *out=NULL;
    PyObject *arr1=NULL, *arr2=NULL, *oarr=NULL;

    if (!PyArg_ParseTuple(args, "OOO!", &arg1, &arg2, &PyArray_Type, &out))
        return NULL;

    arr1 = PyArray_FROM_OTF(arg1, NPY_FLOAT, NPY_IN_ARRAY);
    if (arr1 == NULL)
        return NULL;
    arr2 = PyArray_FROM_OTF(arg2, NPY_FLOAT, NPY_IN_ARRAY);
    if (arr2 == NULL)
        goto fail;
    oarr = PyArray_FROM_OTF(out, NPY_FLOAT, NPY_INOUT_ARRAY);
    if (oarr == NULL)
        goto fail;

    {
        int nd = PyArray_NDIM(arr1);
        assert(nd == 4);
        nd = PyArray_NDIM(arr2);
        assert(nd == 4);
        nd = PyArray_NDIM(oarr);
        assert(nd == 4);

        tensor_4d_input<float> input((float*)PyArray_DATA(arr1), PyArray_STRIDES(arr1), PyArray_DIMS(arr1));
        tensor_4d_kernel<float> kernel((float*)PyArray_DATA(arr2), PyArray_STRIDES(arr2), PyArray_DIMS(arr2));
        tensor_4d_output<float> output((float*)PyArray_DATA(oarr), PyArray_STRIDES(oarr), PyArray_DIMS(oarr));

        conv2d_im2col<float>(input, kernel, output);

        Py_DECREF(arr1);
        Py_DECREF(arr2);
        Py_DECREF(oarr);
        Py_INCREF(Py_None);
        return Py_None;
    }
fail:
    Py_XDECREF(arr1);
    Py_XDECREF(arr2);
    // TODO
    // PyArray_DiscardWritebackIfCopy(oarr);
    Py_XDECREF(oarr);
    return NULL;
}

/*  define functions in module */
static PyMethodDef Methods[] = {
    { "conv2d_im2col", conv2d_im2col_wrapper, METH_VARARGS, "Convolution via Tensor Multiplication" },
    { NULL, NULL, 0, NULL }
};


/* module initialization */
static struct PyModuleDef Module = {
    PyModuleDef_HEAD_INIT,
    "libconv2d", "Collection of 2d convolution routines",
    -1,
    Methods
};

PyMODINIT_FUNC PyInit_libconv2d(void) {
    PyObject* module;
    module = PyModule_Create(&Module);
    if(module==NULL)
        return NULL;
    /* IMPORTANT: this must be called */
    import_array();
    if (PyErr_Occurred())
        return NULL;
    return module;
}

