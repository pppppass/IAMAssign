#include "Exts.h"

// Wrapper of PDE solver using Cholesky decomposition
static PyObject* pde_imp_cho_wrapper(PyObject* self, PyObject* args)
{
    PyObject* data_obj = NULL, * ind_obj = NULL, * ptr_obj = NULL, * init_obj = NULL;
    int width, rep;

    if (!PyArg_ParseTuple(
        args, "O!O!O!O!ii",
        &PyArray_Type, &data_obj,
        &PyArray_Type, &ind_obj,
        &PyArray_Type, &ptr_obj,
        &PyArray_Type, &init_obj,
        &width, &rep
    ))
        return NULL;
    
    PyArrayObject* data_arr = (PyArrayObject*)PyArray_FROM_OTF(data_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* ind_arr = (PyArrayObject*)PyArray_FROM_OTF(ind_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* ptr_arr = (PyArrayObject*)PyArray_FROM_OTF(ptr_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* init_arr = (PyArrayObject*)PyArray_FROM_OTF(init_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
    if (!data_arr || !ind_arr || !ptr_arr || !init_arr)
        return NULL;

    int n = PyArray_SIZE(init_arr);
    double* data_ptr = PyArray_DATA(data_arr);
    int* ind_ptr = PyArray_DATA(ind_arr);
    int* ptr_ptr = PyArray_DATA(ptr_arr);
    double* init_ptr = PyArray_DATA(init_arr);

    int ctr = evolve_cho(n, data_ptr, ind_ptr, ptr_ptr, init_ptr, width, rep);

    Py_DECREF(data_arr);
    Py_DECREF(ind_arr);
    Py_DECREF(ptr_arr);
    PyArray_ResolveWritebackIfCopy(init_arr);
    Py_DECREF(init_arr);
    
    return Py_BuildValue("i", ctr);
}

// Wrapper of PDE solver using Gauss-Seidel
static PyObject* pde_imp_gs_wrapper(PyObject* self, PyObject* args)
{
    PyObject* data_obj = NULL, * ind_obj = NULL, * ptr_obj = NULL, * init_obj = NULL;
    double eps;
    int chk, rep;

    if (!PyArg_ParseTuple(
        args, "O!O!O!O!dii",
        &PyArray_Type, &data_obj,
        &PyArray_Type, &ind_obj,
        &PyArray_Type, &ptr_obj,
        &PyArray_Type, &init_obj,
        &eps, &chk, &rep
    ))
        return NULL;
    
    PyArrayObject* data_arr = (PyArrayObject*)PyArray_FROM_OTF(data_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* ind_arr = (PyArrayObject*)PyArray_FROM_OTF(ind_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* ptr_arr = (PyArrayObject*)PyArray_FROM_OTF(ptr_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* init_arr = (PyArrayObject*)PyArray_FROM_OTF(init_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
    if (!data_arr || !ind_arr || !ptr_arr || !init_arr)
        return NULL;

    int n = PyArray_SIZE(init_arr);
    double* data_ptr = PyArray_DATA(data_arr);
    int* ind_ptr = PyArray_DATA(ind_arr);
    int* ptr_ptr = PyArray_DATA(ptr_arr);
    double* init_ptr = PyArray_DATA(init_arr);

    int ctr = evolve_gs(n, data_ptr, ind_ptr, ptr_ptr, init_ptr, eps, chk, rep);

    Py_DECREF(data_arr);
    Py_DECREF(ind_arr);
    Py_DECREF(ptr_arr);
    PyArray_ResolveWritebackIfCopy(init_arr);
    Py_DECREF(init_arr);
    
    return Py_BuildValue("i", ctr);
}

// Wrapper of PDE solver using steepest descent
static PyObject* pde_imp_sd_wrapper(PyObject* self, PyObject* args)
{
    PyObject* data_obj = NULL, * ind_obj = NULL, * ptr_obj = NULL, * init_obj = NULL;
    double eps;
    int rep;

    if (!PyArg_ParseTuple(
        args, "O!O!O!O!di",
        &PyArray_Type, &data_obj,
        &PyArray_Type, &ind_obj,
        &PyArray_Type, &ptr_obj,
        &PyArray_Type, &init_obj,
        &eps, &rep
    ))
        return NULL;
    
    PyArrayObject* data_arr = (PyArrayObject*)PyArray_FROM_OTF(data_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* ind_arr = (PyArrayObject*)PyArray_FROM_OTF(ind_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* ptr_arr = (PyArrayObject*)PyArray_FROM_OTF(ptr_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* init_arr = (PyArrayObject*)PyArray_FROM_OTF(init_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
    if (!data_arr || !ind_arr || !ptr_arr || !init_arr)
        return NULL;

    int n = PyArray_SIZE(init_arr);
    double* data_ptr = PyArray_DATA(data_arr);
    int* ind_ptr = PyArray_DATA(ind_arr);
    int* ptr_ptr = PyArray_DATA(ptr_arr);
    double* init_ptr = PyArray_DATA(init_arr);

    int ctr = evolve_sd(n, data_ptr, ind_ptr, ptr_ptr, init_ptr, eps, rep);

    Py_DECREF(data_arr);
    Py_DECREF(ind_arr);
    Py_DECREF(ptr_arr);
    PyArray_ResolveWritebackIfCopy(init_arr);
    Py_DECREF(init_arr);
    
    return Py_BuildValue("i", ctr);
}

// Wrapper of PDE solver using conjugate descent
static PyObject* pde_imp_cg_wrapper(PyObject* self, PyObject* args)
{
    PyObject* data_obj, * ind_obj, * ptr_obj, * init_obj;
    double eps;
    int rep;

    if (!PyArg_ParseTuple(
        args, "O!O!O!O!di",
        &PyArray_Type, &data_obj,
        &PyArray_Type, &ind_obj,
        &PyArray_Type, &ptr_obj,
        &PyArray_Type, &init_obj,
        &eps, &rep
    ))
        return NULL;
    
    PyArrayObject* data_arr = (PyArrayObject*)PyArray_FROM_OTF(data_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* ind_arr = (PyArrayObject*)PyArray_FROM_OTF(ind_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* ptr_arr = (PyArrayObject*)PyArray_FROM_OTF(ptr_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* init_arr = (PyArrayObject*)PyArray_FROM_OTF(init_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!data_arr || !ind_arr || !ptr_arr || !init_arr)
        return NULL;

    int n = PyArray_SIZE(init_arr);
    double* data_ptr = PyArray_DATA(data_arr);
    int* ind_ptr = PyArray_DATA(ind_arr);
    int* ptr_ptr = PyArray_DATA(ptr_arr);
    double* init_ptr = PyArray_DATA(init_arr);

    int ctr = evolve_cg(n, data_ptr, ind_ptr, ptr_ptr, init_ptr, eps, rep);

    Py_DECREF(data_arr);
    Py_DECREF(ind_arr);
    Py_DECREF(ptr_arr);
    PyArray_ResolveWritebackIfCopy(init_arr);
    Py_DECREF(init_arr);
    
    return Py_BuildValue("i", ctr);
}

// Wrapper of PDE solver using multigrid
static PyObject* pde_imp_mg_wrapper(PyObject* self, PyObject* args)
{
    PyObject* data_list_obj, * ind_list_obj, * ptr_list_obj, * init_obj, * num_obj;
    double eps;
    int size, chk, rep;
    
    if (!PyArg_ParseTuple(
        args, "iO!O!O!O!O!dii",
        &size,
        &PyList_Type, &data_list_obj,
        &PyList_Type, &ind_list_obj,
        &PyList_Type, &ptr_list_obj,
        &PyArray_Type, &init_obj,
        &PyArray_Type, &num_obj,
        &eps, &chk, &rep
    ))
        return NULL;
    
    PyArrayObject* init_arr = (PyArrayObject*)PyArray_FROM_OTF(init_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
    PyArrayObject* num_arr = (PyArrayObject*)PyArray_FROM_OTF(num_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    if (!init_arr || !num_arr)
        return NULL;

    int depth = PyArray_SIZE(num_arr);
    double* init_ptr = PyArray_DATA(init_arr);
    int* num_ptr = PyArray_DATA(num_arr);

    PyArrayObject** data_arrs, ** ind_arrs, ** ptr_arrs;
    data_arrs = malloc(depth * sizeof(PyArrayObject*)), ind_arrs = malloc(depth * sizeof(PyArrayObject*)), ptr_arrs = malloc(depth * sizeof(PyArrayObject*));
    double** data_ptrs;
    int ** ind_ptrs, **ptr_ptrs;
    data_ptrs = malloc(depth * sizeof(double*)), ind_ptrs = malloc(depth * sizeof(int*)), ptr_ptrs = malloc(depth * sizeof(int*));
    for (int i = 0; i < depth; i++)
    {
        PyObject* data_obj = PyList_GetItem(data_list_obj, i), * ind_obj = PyList_GetItem(ind_list_obj, i), * ptr_obj = PyList_GetItem(ptr_list_obj, i);
        if (!data_obj || !ind_obj || !ptr_obj)
            return NULL;
        data_arrs[i] = (PyArrayObject*)PyArray_FROM_OTF(data_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        ind_arrs[i] = (PyArrayObject*)PyArray_FROM_OTF(ind_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
        ptr_arrs[i] = (PyArrayObject*)PyArray_FROM_OTF(ptr_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
        if (!data_arrs[i] || !ind_arrs[i] || !ptr_arrs[i])
            return NULL;
        data_ptrs[i] = PyArray_DATA(data_arrs[i]);
        ind_ptrs[i] = PyArray_DATA(ind_arrs[i]);
        ptr_ptrs[i] = PyArray_DATA(ptr_arrs[i]);
    }

    int ctr = evolve_mg(size, data_ptrs, ind_ptrs, ptr_ptrs, init_ptr, num_ptr, depth, eps, chk, rep);

    for (int i = 0; i < depth; i++)
    {
        Py_DECREF(data_arrs[i]);
        Py_DECREF(ind_arrs[i]);
        Py_DECREF(ptr_arrs[i]);
    }
    free(data_ptrs), free(ind_ptrs), free(ptr_ptrs);
    free(data_arrs), free(ind_arrs), free(ptr_arrs);
    Py_DECREF(num_arr);
    PyArray_ResolveWritebackIfCopy(init_arr);
    Py_DECREF(init_arr);

    return Py_BuildValue("i", ctr);
}

static PyMethodDef methods[] = 
{
    {"pde_imp_cho_wrapper", pde_imp_cho_wrapper, METH_VARARGS, NULL},
    {"pde_imp_gs_wrapper", pde_imp_gs_wrapper, METH_VARARGS, NULL},
    {"pde_imp_sd_wrapper", pde_imp_sd_wrapper, METH_VARARGS, NULL},
    {"pde_imp_cg_wrapper", pde_imp_cg_wrapper, METH_VARARGS, NULL},
    {"pde_imp_mg_wrapper", pde_imp_mg_wrapper, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef table = 
{
    PyModuleDef_HEAD_INIT,
    "exts", NULL, -1, methods
};

PyMODINIT_FUNC PyInit_exts(void)
{
    import_array();
    return PyModule_Create(&table);
}
