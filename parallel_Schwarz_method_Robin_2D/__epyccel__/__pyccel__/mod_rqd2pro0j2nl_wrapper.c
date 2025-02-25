#define PY_ARRAY_UNIQUE_SYMBOL CWRAPPER_ARRAY_API
#define MOD_RQD2PRO0J2NL_WRAPPER

#include "mod_rqd2pro0j2nl_wrapper.h"
#include <stdlib.h>
#include <stdint.h>
#include "ndarrays.h"


/*........................................*/


/*........................................*/

/*........................................*/
static PyObject* bind_c_assemble_vector_ex01_wrapper(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject* ne1_obj;
    PyObject* ne2_obj;
    PyObject* ne3_obj;
    PyObject* ne4_obj;
    PyObject* p1_obj;
    PyObject* p2_obj;
    PyObject* p3_obj;
    PyObject* p4_obj;
    PyObject* spans_1_obj;
    PyObject* spans_2_obj;
    PyObject* spans_3_obj;
    PyObject* spans_4_obj;
    PyObject* basis_1_obj;
    PyObject* basis_2_obj;
    PyObject* basis_3_obj;
    PyObject* basis_4_obj;
    PyObject* weights_1_obj;
    PyObject* weights_2_obj;
    PyObject* weights_3_obj;
    PyObject* weights_4_obj;
    PyObject* points_1_obj;
    PyObject* points_2_obj;
    PyObject* points_3_obj;
    PyObject* points_4_obj;
    PyObject* knots_1_obj;
    PyObject* knots_2_obj;
    PyObject* knots_3_obj;
    PyObject* knots_4_obj;
    PyObject* vector_d_obj;
    PyObject* ovlp_value_obj;
    PyObject* S_DDM_obj;
    PyObject* domain_nb_obj;
    PyObject* rhs_obj;
    int64_t ne1;
    int64_t ne2;
    int64_t ne3;
    int64_t ne4;
    int64_t p1;
    int64_t p2;
    int64_t p3;
    int64_t p4;
    t_ndarray spans_1 = {.shape = NULL};
    void* bound_spans_1;
    int64_t bound_spans_1_shape_1;
    int64_t bound_spans_1_stride_1;
    t_ndarray spans_2 = {.shape = NULL};
    void* bound_spans_2;
    int64_t bound_spans_2_shape_1;
    int64_t bound_spans_2_stride_1;
    t_ndarray spans_3 = {.shape = NULL};
    void* bound_spans_3;
    int64_t bound_spans_3_shape_1;
    int64_t bound_spans_3_stride_1;
    t_ndarray spans_4 = {.shape = NULL};
    void* bound_spans_4;
    int64_t bound_spans_4_shape_1;
    int64_t bound_spans_4_stride_1;
    t_ndarray basis_1 = {.shape = NULL};
    void* bound_basis_1;
    int64_t bound_basis_1_shape_1;
    int64_t bound_basis_1_shape_2;
    int64_t bound_basis_1_shape_3;
    int64_t bound_basis_1_shape_4;
    int64_t bound_basis_1_stride_1;
    int64_t bound_basis_1_stride_2;
    int64_t bound_basis_1_stride_3;
    int64_t bound_basis_1_stride_4;
    t_ndarray basis_2 = {.shape = NULL};
    void* bound_basis_2;
    int64_t bound_basis_2_shape_1;
    int64_t bound_basis_2_shape_2;
    int64_t bound_basis_2_shape_3;
    int64_t bound_basis_2_shape_4;
    int64_t bound_basis_2_stride_1;
    int64_t bound_basis_2_stride_2;
    int64_t bound_basis_2_stride_3;
    int64_t bound_basis_2_stride_4;
    t_ndarray basis_3 = {.shape = NULL};
    void* bound_basis_3;
    int64_t bound_basis_3_shape_1;
    int64_t bound_basis_3_shape_2;
    int64_t bound_basis_3_shape_3;
    int64_t bound_basis_3_shape_4;
    int64_t bound_basis_3_stride_1;
    int64_t bound_basis_3_stride_2;
    int64_t bound_basis_3_stride_3;
    int64_t bound_basis_3_stride_4;
    t_ndarray basis_4 = {.shape = NULL};
    void* bound_basis_4;
    int64_t bound_basis_4_shape_1;
    int64_t bound_basis_4_shape_2;
    int64_t bound_basis_4_shape_3;
    int64_t bound_basis_4_shape_4;
    int64_t bound_basis_4_stride_1;
    int64_t bound_basis_4_stride_2;
    int64_t bound_basis_4_stride_3;
    int64_t bound_basis_4_stride_4;
    t_ndarray weights_1 = {.shape = NULL};
    void* bound_weights_1;
    int64_t bound_weights_1_shape_1;
    int64_t bound_weights_1_shape_2;
    int64_t bound_weights_1_stride_1;
    int64_t bound_weights_1_stride_2;
    t_ndarray weights_2 = {.shape = NULL};
    void* bound_weights_2;
    int64_t bound_weights_2_shape_1;
    int64_t bound_weights_2_shape_2;
    int64_t bound_weights_2_stride_1;
    int64_t bound_weights_2_stride_2;
    t_ndarray weights_3 = {.shape = NULL};
    void* bound_weights_3;
    int64_t bound_weights_3_shape_1;
    int64_t bound_weights_3_shape_2;
    int64_t bound_weights_3_stride_1;
    int64_t bound_weights_3_stride_2;
    t_ndarray weights_4 = {.shape = NULL};
    void* bound_weights_4;
    int64_t bound_weights_4_shape_1;
    int64_t bound_weights_4_shape_2;
    int64_t bound_weights_4_stride_1;
    int64_t bound_weights_4_stride_2;
    t_ndarray points_1 = {.shape = NULL};
    void* bound_points_1;
    int64_t bound_points_1_shape_1;
    int64_t bound_points_1_shape_2;
    int64_t bound_points_1_stride_1;
    int64_t bound_points_1_stride_2;
    t_ndarray points_2 = {.shape = NULL};
    void* bound_points_2;
    int64_t bound_points_2_shape_1;
    int64_t bound_points_2_shape_2;
    int64_t bound_points_2_stride_1;
    int64_t bound_points_2_stride_2;
    t_ndarray points_3 = {.shape = NULL};
    void* bound_points_3;
    int64_t bound_points_3_shape_1;
    int64_t bound_points_3_shape_2;
    int64_t bound_points_3_stride_1;
    int64_t bound_points_3_stride_2;
    t_ndarray points_4 = {.shape = NULL};
    void* bound_points_4;
    int64_t bound_points_4_shape_1;
    int64_t bound_points_4_shape_2;
    int64_t bound_points_4_stride_1;
    int64_t bound_points_4_stride_2;
    t_ndarray knots_1 = {.shape = NULL};
    void* bound_knots_1;
    int64_t bound_knots_1_shape_1;
    int64_t bound_knots_1_stride_1;
    t_ndarray knots_2 = {.shape = NULL};
    void* bound_knots_2;
    int64_t bound_knots_2_shape_1;
    int64_t bound_knots_2_stride_1;
    t_ndarray knots_3 = {.shape = NULL};
    void* bound_knots_3;
    int64_t bound_knots_3_shape_1;
    int64_t bound_knots_3_stride_1;
    t_ndarray knots_4 = {.shape = NULL};
    void* bound_knots_4;
    int64_t bound_knots_4_shape_1;
    int64_t bound_knots_4_stride_1;
    t_ndarray vector_d = {.shape = NULL};
    void* bound_vector_d;
    int64_t bound_vector_d_shape_1;
    int64_t bound_vector_d_shape_2;
    int64_t bound_vector_d_stride_1;
    int64_t bound_vector_d_stride_2;
    double ovlp_value;
    double S_DDM;
    int64_t domain_nb;
    t_ndarray rhs = {.shape = NULL};
    void* bound_rhs;
    int64_t bound_rhs_shape_1;
    int64_t bound_rhs_shape_2;
    int64_t bound_rhs_stride_1;
    int64_t bound_rhs_stride_2;
    static char *kwlist[] = {
        "ne1",
        "ne2",
        "ne3",
        "ne4",
        "p1",
        "p2",
        "p3",
        "p4",
        "spans_1",
        "spans_2",
        "spans_3",
        "spans_4",
        "basis_1",
        "basis_2",
        "basis_3",
        "basis_4",
        "weights_1",
        "weights_2",
        "weights_3",
        "weights_4",
        "points_1",
        "points_2",
        "points_3",
        "points_4",
        "knots_1",
        "knots_2",
        "knots_3",
        "knots_4",
        "vector_d",
        "ovlp_value",
        "S_DDM",
        "domain_nb",
        "rhs",
        NULL
    };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO", kwlist, &ne1_obj, &ne2_obj, &ne3_obj, &ne4_obj, &p1_obj, &p2_obj, &p3_obj, &p4_obj, &spans_1_obj, &spans_2_obj, &spans_3_obj, &spans_4_obj, &basis_1_obj, &basis_2_obj, &basis_3_obj, &basis_4_obj, &weights_1_obj, &weights_2_obj, &weights_3_obj, &weights_4_obj, &points_1_obj, &points_2_obj, &points_3_obj, &points_4_obj, &knots_1_obj, &knots_2_obj, &knots_3_obj, &knots_4_obj, &vector_d_obj, &ovlp_value_obj, &S_DDM_obj, &domain_nb_obj, &rhs_obj))
    {
        return NULL;
    }
    if (PyIs_NativeInt(ne1_obj))
    {
        ne1 = PyInt64_to_Int64(ne1_obj);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Expected an argument of type int for argument ne1");
        return NULL;
    }
    if (PyIs_NativeInt(ne2_obj))
    {
        ne2 = PyInt64_to_Int64(ne2_obj);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Expected an argument of type int for argument ne2");
        return NULL;
    }
    if (PyIs_NativeInt(ne3_obj))
    {
        ne3 = PyInt64_to_Int64(ne3_obj);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Expected an argument of type int for argument ne3");
        return NULL;
    }
    if (PyIs_NativeInt(ne4_obj))
    {
        ne4 = PyInt64_to_Int64(ne4_obj);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Expected an argument of type int for argument ne4");
        return NULL;
    }
    if (PyIs_NativeInt(p1_obj))
    {
        p1 = PyInt64_to_Int64(p1_obj);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Expected an argument of type int for argument p1");
        return NULL;
    }
    if (PyIs_NativeInt(p2_obj))
    {
        p2 = PyInt64_to_Int64(p2_obj);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Expected an argument of type int for argument p2");
        return NULL;
    }
    if (PyIs_NativeInt(p3_obj))
    {
        p3 = PyInt64_to_Int64(p3_obj);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Expected an argument of type int for argument p3");
        return NULL;
    }
    if (PyIs_NativeInt(p4_obj))
    {
        p4 = PyInt64_to_Int64(p4_obj);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Expected an argument of type int for argument p4");
        return NULL;
    }
    if (pyarray_check(spans_1_obj, NPY_LONG, INT64_C(1), NO_ORDER_CHECK))
    {
        spans_1 = pyarray_to_ndarray(spans_1_obj);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Expected an argument of type int for argument spans_1");
        return NULL;
    }
    bound_spans_1 = nd_data(&spans_1);
    bound_spans_1_shape_1 = nd_ndim(&spans_1, INT64_C(0));
    bound_spans_1_stride_1 = nd_nstep_F(&spans_1, INT64_C(0));
    if (pyarray_check(spans_2_obj, NPY_LONG, INT64_C(1), NO_ORDER_CHECK))
    {
        spans_2 = pyarray_to_ndarray(spans_2_obj);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Expected an argument of type int for argument spans_2");
        return NULL;
    }
    bound_spans_2 = nd_data(&spans_2);
    bound_spans_2_shape_1 = nd_ndim(&spans_2, INT64_C(0));
    bound_spans_2_stride_1 = nd_nstep_F(&spans_2, INT64_C(0));
    if (pyarray_check(spans_3_obj, NPY_LONG, INT64_C(1), NO_ORDER_CHECK))
    {
        spans_3 = pyarray_to_ndarray(spans_3_obj);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Expected an argument of type int for argument spans_3");
        return NULL;
    }
    bound_spans_3 = nd_data(&spans_3);
    bound_spans_3_shape_1 = nd_ndim(&spans_3, INT64_C(0));
    bound_spans_3_stride_1 = nd_nstep_F(&spans_3, INT64_C(0));
    if (pyarray_check(spans_4_obj, NPY_LONG, INT64_C(1), NO_ORDER_CHECK))
    {
        spans_4 = pyarray_to_ndarray(spans_4_obj);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Expected an argument of type int for argument spans_4");
        return NULL;
    }
    bound_spans_4 = nd_data(&spans_4);
    bound_spans_4_shape_1 = nd_ndim(&spans_4, INT64_C(0));
    bound_spans_4_stride_1 = nd_nstep_F(&spans_4, INT64_C(0));
    if (pyarray_check(basis_1_obj, NPY_DOUBLE, INT64_C(4), NPY_ARRAY_C_CONTIGUOUS))
    {
        basis_1 = pyarray_to_ndarray(basis_1_obj);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Expected an argument of type float for argument basis_1");
        return NULL;
    }
    bound_basis_1 = nd_data(&basis_1);
    bound_basis_1_shape_1 = nd_ndim(&basis_1, INT64_C(0));
    bound_basis_1_shape_2 = nd_ndim(&basis_1, INT64_C(1));
    bound_basis_1_shape_3 = nd_ndim(&basis_1, INT64_C(2));
    bound_basis_1_shape_4 = nd_ndim(&basis_1, INT64_C(3));
    bound_basis_1_stride_1 = nd_nstep_C(&basis_1, INT64_C(0));
    bound_basis_1_stride_2 = nd_nstep_C(&basis_1, INT64_C(1));
    bound_basis_1_stride_3 = nd_nstep_C(&basis_1, INT64_C(2));
    bound_basis_1_stride_4 = nd_nstep_C(&basis_1, INT64_C(3));
    if (pyarray_check(basis_2_obj, NPY_DOUBLE, INT64_C(4), NPY_ARRAY_C_CONTIGUOUS))
    {
        basis_2 = pyarray_to_ndarray(basis_2_obj);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Expected an argument of type float for argument basis_2");
        return NULL;
    }
    bound_basis_2 = nd_data(&basis_2);
    bound_basis_2_shape_1 = nd_ndim(&basis_2, INT64_C(0));
    bound_basis_2_shape_2 = nd_ndim(&basis_2, INT64_C(1));
    bound_basis_2_shape_3 = nd_ndim(&basis_2, INT64_C(2));
    bound_basis_2_shape_4 = nd_ndim(&basis_2, INT64_C(3));
    bound_basis_2_stride_1 = nd_nstep_C(&basis_2, INT64_C(0));
    bound_basis_2_stride_2 = nd_nstep_C(&basis_2, INT64_C(1));
    bound_basis_2_stride_3 = nd_nstep_C(&basis_2, INT64_C(2));
    bound_basis_2_stride_4 = nd_nstep_C(&basis_2, INT64_C(3));
    if (pyarray_check(basis_3_obj, NPY_DOUBLE, INT64_C(4), NPY_ARRAY_C_CONTIGUOUS))
    {
        basis_3 = pyarray_to_ndarray(basis_3_obj);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Expected an argument of type float for argument basis_3");
        return NULL;
    }
    bound_basis_3 = nd_data(&basis_3);
    bound_basis_3_shape_1 = nd_ndim(&basis_3, INT64_C(0));
    bound_basis_3_shape_2 = nd_ndim(&basis_3, INT64_C(1));
    bound_basis_3_shape_3 = nd_ndim(&basis_3, INT64_C(2));
    bound_basis_3_shape_4 = nd_ndim(&basis_3, INT64_C(3));
    bound_basis_3_stride_1 = nd_nstep_C(&basis_3, INT64_C(0));
    bound_basis_3_stride_2 = nd_nstep_C(&basis_3, INT64_C(1));
    bound_basis_3_stride_3 = nd_nstep_C(&basis_3, INT64_C(2));
    bound_basis_3_stride_4 = nd_nstep_C(&basis_3, INT64_C(3));
    if (pyarray_check(basis_4_obj, NPY_DOUBLE, INT64_C(4), NPY_ARRAY_C_CONTIGUOUS))
    {
        basis_4 = pyarray_to_ndarray(basis_4_obj);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Expected an argument of type float for argument basis_4");
        return NULL;
    }
    bound_basis_4 = nd_data(&basis_4);
    bound_basis_4_shape_1 = nd_ndim(&basis_4, INT64_C(0));
    bound_basis_4_shape_2 = nd_ndim(&basis_4, INT64_C(1));
    bound_basis_4_shape_3 = nd_ndim(&basis_4, INT64_C(2));
    bound_basis_4_shape_4 = nd_ndim(&basis_4, INT64_C(3));
    bound_basis_4_stride_1 = nd_nstep_C(&basis_4, INT64_C(0));
    bound_basis_4_stride_2 = nd_nstep_C(&basis_4, INT64_C(1));
    bound_basis_4_stride_3 = nd_nstep_C(&basis_4, INT64_C(2));
    bound_basis_4_stride_4 = nd_nstep_C(&basis_4, INT64_C(3));
    if (pyarray_check(weights_1_obj, NPY_DOUBLE, INT64_C(2), NPY_ARRAY_C_CONTIGUOUS))
    {
        weights_1 = pyarray_to_ndarray(weights_1_obj);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Expected an argument of type float for argument weights_1");
        return NULL;
    }
    bound_weights_1 = nd_data(&weights_1);
    bound_weights_1_shape_1 = nd_ndim(&weights_1, INT64_C(0));
    bound_weights_1_shape_2 = nd_ndim(&weights_1, INT64_C(1));
    bound_weights_1_stride_1 = nd_nstep_C(&weights_1, INT64_C(0));
    bound_weights_1_stride_2 = nd_nstep_C(&weights_1, INT64_C(1));
    if (pyarray_check(weights_2_obj, NPY_DOUBLE, INT64_C(2), NPY_ARRAY_C_CONTIGUOUS))
    {
        weights_2 = pyarray_to_ndarray(weights_2_obj);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Expected an argument of type float for argument weights_2");
        return NULL;
    }
    bound_weights_2 = nd_data(&weights_2);
    bound_weights_2_shape_1 = nd_ndim(&weights_2, INT64_C(0));
    bound_weights_2_shape_2 = nd_ndim(&weights_2, INT64_C(1));
    bound_weights_2_stride_1 = nd_nstep_C(&weights_2, INT64_C(0));
    bound_weights_2_stride_2 = nd_nstep_C(&weights_2, INT64_C(1));
    if (pyarray_check(weights_3_obj, NPY_DOUBLE, INT64_C(2), NPY_ARRAY_C_CONTIGUOUS))
    {
        weights_3 = pyarray_to_ndarray(weights_3_obj);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Expected an argument of type float for argument weights_3");
        return NULL;
    }
    bound_weights_3 = nd_data(&weights_3);
    bound_weights_3_shape_1 = nd_ndim(&weights_3, INT64_C(0));
    bound_weights_3_shape_2 = nd_ndim(&weights_3, INT64_C(1));
    bound_weights_3_stride_1 = nd_nstep_C(&weights_3, INT64_C(0));
    bound_weights_3_stride_2 = nd_nstep_C(&weights_3, INT64_C(1));
    if (pyarray_check(weights_4_obj, NPY_DOUBLE, INT64_C(2), NPY_ARRAY_C_CONTIGUOUS))
    {
        weights_4 = pyarray_to_ndarray(weights_4_obj);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Expected an argument of type float for argument weights_4");
        return NULL;
    }
    bound_weights_4 = nd_data(&weights_4);
    bound_weights_4_shape_1 = nd_ndim(&weights_4, INT64_C(0));
    bound_weights_4_shape_2 = nd_ndim(&weights_4, INT64_C(1));
    bound_weights_4_stride_1 = nd_nstep_C(&weights_4, INT64_C(0));
    bound_weights_4_stride_2 = nd_nstep_C(&weights_4, INT64_C(1));
    if (pyarray_check(points_1_obj, NPY_DOUBLE, INT64_C(2), NPY_ARRAY_C_CONTIGUOUS))
    {
        points_1 = pyarray_to_ndarray(points_1_obj);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Expected an argument of type float for argument points_1");
        return NULL;
    }
    bound_points_1 = nd_data(&points_1);
    bound_points_1_shape_1 = nd_ndim(&points_1, INT64_C(0));
    bound_points_1_shape_2 = nd_ndim(&points_1, INT64_C(1));
    bound_points_1_stride_1 = nd_nstep_C(&points_1, INT64_C(0));
    bound_points_1_stride_2 = nd_nstep_C(&points_1, INT64_C(1));
    if (pyarray_check(points_2_obj, NPY_DOUBLE, INT64_C(2), NPY_ARRAY_C_CONTIGUOUS))
    {
        points_2 = pyarray_to_ndarray(points_2_obj);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Expected an argument of type float for argument points_2");
        return NULL;
    }
    bound_points_2 = nd_data(&points_2);
    bound_points_2_shape_1 = nd_ndim(&points_2, INT64_C(0));
    bound_points_2_shape_2 = nd_ndim(&points_2, INT64_C(1));
    bound_points_2_stride_1 = nd_nstep_C(&points_2, INT64_C(0));
    bound_points_2_stride_2 = nd_nstep_C(&points_2, INT64_C(1));
    if (pyarray_check(points_3_obj, NPY_DOUBLE, INT64_C(2), NPY_ARRAY_C_CONTIGUOUS))
    {
        points_3 = pyarray_to_ndarray(points_3_obj);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Expected an argument of type float for argument points_3");
        return NULL;
    }
    bound_points_3 = nd_data(&points_3);
    bound_points_3_shape_1 = nd_ndim(&points_3, INT64_C(0));
    bound_points_3_shape_2 = nd_ndim(&points_3, INT64_C(1));
    bound_points_3_stride_1 = nd_nstep_C(&points_3, INT64_C(0));
    bound_points_3_stride_2 = nd_nstep_C(&points_3, INT64_C(1));
    if (pyarray_check(points_4_obj, NPY_DOUBLE, INT64_C(2), NPY_ARRAY_C_CONTIGUOUS))
    {
        points_4 = pyarray_to_ndarray(points_4_obj);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Expected an argument of type float for argument points_4");
        return NULL;
    }
    bound_points_4 = nd_data(&points_4);
    bound_points_4_shape_1 = nd_ndim(&points_4, INT64_C(0));
    bound_points_4_shape_2 = nd_ndim(&points_4, INT64_C(1));
    bound_points_4_stride_1 = nd_nstep_C(&points_4, INT64_C(0));
    bound_points_4_stride_2 = nd_nstep_C(&points_4, INT64_C(1));
    if (pyarray_check(knots_1_obj, NPY_DOUBLE, INT64_C(1), NO_ORDER_CHECK))
    {
        knots_1 = pyarray_to_ndarray(knots_1_obj);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Expected an argument of type float for argument knots_1");
        return NULL;
    }
    bound_knots_1 = nd_data(&knots_1);
    bound_knots_1_shape_1 = nd_ndim(&knots_1, INT64_C(0));
    bound_knots_1_stride_1 = nd_nstep_F(&knots_1, INT64_C(0));
    if (pyarray_check(knots_2_obj, NPY_DOUBLE, INT64_C(1), NO_ORDER_CHECK))
    {
        knots_2 = pyarray_to_ndarray(knots_2_obj);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Expected an argument of type float for argument knots_2");
        return NULL;
    }
    bound_knots_2 = nd_data(&knots_2);
    bound_knots_2_shape_1 = nd_ndim(&knots_2, INT64_C(0));
    bound_knots_2_stride_1 = nd_nstep_F(&knots_2, INT64_C(0));
    if (pyarray_check(knots_3_obj, NPY_DOUBLE, INT64_C(1), NO_ORDER_CHECK))
    {
        knots_3 = pyarray_to_ndarray(knots_3_obj);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Expected an argument of type float for argument knots_3");
        return NULL;
    }
    bound_knots_3 = nd_data(&knots_3);
    bound_knots_3_shape_1 = nd_ndim(&knots_3, INT64_C(0));
    bound_knots_3_stride_1 = nd_nstep_F(&knots_3, INT64_C(0));
    if (pyarray_check(knots_4_obj, NPY_DOUBLE, INT64_C(1), NO_ORDER_CHECK))
    {
        knots_4 = pyarray_to_ndarray(knots_4_obj);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Expected an argument of type float for argument knots_4");
        return NULL;
    }
    bound_knots_4 = nd_data(&knots_4);
    bound_knots_4_shape_1 = nd_ndim(&knots_4, INT64_C(0));
    bound_knots_4_stride_1 = nd_nstep_F(&knots_4, INT64_C(0));
    if (pyarray_check(vector_d_obj, NPY_DOUBLE, INT64_C(2), NPY_ARRAY_C_CONTIGUOUS))
    {
        vector_d = pyarray_to_ndarray(vector_d_obj);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Expected an argument of type float for argument vector_d");
        return NULL;
    }
    bound_vector_d = nd_data(&vector_d);
    bound_vector_d_shape_1 = nd_ndim(&vector_d, INT64_C(0));
    bound_vector_d_shape_2 = nd_ndim(&vector_d, INT64_C(1));
    bound_vector_d_stride_1 = nd_nstep_C(&vector_d, INT64_C(0));
    bound_vector_d_stride_2 = nd_nstep_C(&vector_d, INT64_C(1));
    if (PyIs_NativeFloat(ovlp_value_obj))
    {
        ovlp_value = PyDouble_to_Double(ovlp_value_obj);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Expected an argument of type float for argument ovlp_value");
        return NULL;
    }
    if (PyIs_NativeFloat(S_DDM_obj))
    {
        S_DDM = PyDouble_to_Double(S_DDM_obj);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Expected an argument of type float for argument S_DDM");
        return NULL;
    }
    if (PyIs_NativeInt(domain_nb_obj))
    {
        domain_nb = PyInt64_to_Int64(domain_nb_obj);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Expected an argument of type int for argument domain_nb");
        return NULL;
    }
    if (pyarray_check(rhs_obj, NPY_DOUBLE, INT64_C(2), NPY_ARRAY_C_CONTIGUOUS))
    {
        rhs = pyarray_to_ndarray(rhs_obj);
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "Expected an argument of type float for argument rhs");
        return NULL;
    }
    bound_rhs = nd_data(&rhs);
    bound_rhs_shape_1 = nd_ndim(&rhs, INT64_C(0));
    bound_rhs_shape_2 = nd_ndim(&rhs, INT64_C(1));
    bound_rhs_stride_1 = nd_nstep_C(&rhs, INT64_C(0));
    bound_rhs_stride_2 = nd_nstep_C(&rhs, INT64_C(1));
    bind_c_assemble_vector_ex01(ne1, ne2, ne3, ne4, p1, p2, p3, p4, bound_spans_1, bound_spans_1_shape_1, bound_spans_1_stride_1, bound_spans_2, bound_spans_2_shape_1, bound_spans_2_stride_1, bound_spans_3, bound_spans_3_shape_1, bound_spans_3_stride_1, bound_spans_4, bound_spans_4_shape_1, bound_spans_4_stride_1, bound_basis_1, bound_basis_1_shape_1, bound_basis_1_shape_2, bound_basis_1_shape_3, bound_basis_1_shape_4, bound_basis_1_stride_1, bound_basis_1_stride_2, bound_basis_1_stride_3, bound_basis_1_stride_4, bound_basis_2, bound_basis_2_shape_1, bound_basis_2_shape_2, bound_basis_2_shape_3, bound_basis_2_shape_4, bound_basis_2_stride_1, bound_basis_2_stride_2, bound_basis_2_stride_3, bound_basis_2_stride_4, bound_basis_3, bound_basis_3_shape_1, bound_basis_3_shape_2, bound_basis_3_shape_3, bound_basis_3_shape_4, bound_basis_3_stride_1, bound_basis_3_stride_2, bound_basis_3_stride_3, bound_basis_3_stride_4, bound_basis_4, bound_basis_4_shape_1, bound_basis_4_shape_2, bound_basis_4_shape_3, bound_basis_4_shape_4, bound_basis_4_stride_1, bound_basis_4_stride_2, bound_basis_4_stride_3, bound_basis_4_stride_4, bound_weights_1, bound_weights_1_shape_1, bound_weights_1_shape_2, bound_weights_1_stride_1, bound_weights_1_stride_2, bound_weights_2, bound_weights_2_shape_1, bound_weights_2_shape_2, bound_weights_2_stride_1, bound_weights_2_stride_2, bound_weights_3, bound_weights_3_shape_1, bound_weights_3_shape_2, bound_weights_3_stride_1, bound_weights_3_stride_2, bound_weights_4, bound_weights_4_shape_1, bound_weights_4_shape_2, bound_weights_4_stride_1, bound_weights_4_stride_2, bound_points_1, bound_points_1_shape_1, bound_points_1_shape_2, bound_points_1_stride_1, bound_points_1_stride_2, bound_points_2, bound_points_2_shape_1, bound_points_2_shape_2, bound_points_2_stride_1, bound_points_2_stride_2, bound_points_3, bound_points_3_shape_1, bound_points_3_shape_2, bound_points_3_stride_1, bound_points_3_stride_2, bound_points_4, bound_points_4_shape_1, bound_points_4_shape_2, bound_points_4_stride_1, bound_points_4_stride_2, bound_knots_1, bound_knots_1_shape_1, bound_knots_1_stride_1, bound_knots_2, bound_knots_2_shape_1, bound_knots_2_stride_1, bound_knots_3, bound_knots_3_shape_1, bound_knots_3_stride_1, bound_knots_4, bound_knots_4_shape_1, bound_knots_4_stride_1, bound_vector_d, bound_vector_d_shape_1, bound_vector_d_shape_2, bound_vector_d_stride_1, bound_vector_d_stride_2, ovlp_value, S_DDM, domain_nb, bound_rhs, bound_rhs_shape_1, bound_rhs_shape_2, bound_rhs_stride_1, bound_rhs_stride_2);
    free_pointer(&spans_1);
    free_pointer(&spans_2);
    free_pointer(&spans_3);
    free_pointer(&spans_4);
    free_pointer(&basis_1);
    free_pointer(&basis_2);
    free_pointer(&basis_3);
    free_pointer(&basis_4);
    free_pointer(&weights_1);
    free_pointer(&weights_2);
    free_pointer(&weights_3);
    free_pointer(&weights_4);
    free_pointer(&points_1);
    free_pointer(&points_2);
    free_pointer(&points_3);
    free_pointer(&points_4);
    free_pointer(&knots_1);
    free_pointer(&knots_2);
    free_pointer(&knots_3);
    free_pointer(&knots_4);
    free_pointer(&vector_d);
    free_pointer(&rhs);
    Py_INCREF(Py_None);
    return Py_None;
}
/*........................................*/

/*........................................*/

static PyMethodDef mod_rqd2pro0j2nl_methods[] = {
    {
        "assemble_vector_ex01",
        (PyCFunction)bind_c_assemble_vector_ex01_wrapper,
        METH_VARARGS | METH_KEYWORDS,
        ""
    },
    { NULL, NULL, 0, NULL}
};

/*........................................*/

static struct PyModuleDef mod_rqd2pro0j2nl_module = {
    PyModuleDef_HEAD_INIT,
    /* name of module */
    "mod_rqd2pro0j2nl",
    /* module documentation, may be NULL */
    NULL,
    /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    0,
    mod_rqd2pro0j2nl_methods,
};

/*........................................*/

PyMODINIT_FUNC PyInit_mod_rqd2pro0j2nl(void)
{
    PyObject* mod;
    static void* Pymod_rqd2pro0j2nl_API[0];
    PyObject* c_api_object_0001;
    mod = PyModule_Create(&mod_rqd2pro0j2nl_module);
    if (mod == NULL)
    {
        return NULL;
    }
    c_api_object_0001 = PyCapsule_New((void *)Pymod_rqd2pro0j2nl_API, "mod_rqd2pro0j2nl._C_API", NULL);
    if (PyModule_AddObject(mod, "_C_API", c_api_object_0001) < INT64_C(0))
    {
        Py_DECREF(mod);
        return NULL;
    }
    Py_INCREF(c_api_object_0001);
    import_array();
    return mod;
}
