#ifndef MOD_N1F9SNG7JW4B_WRAPPER_H
#define MOD_N1F9SNG7JW4B_WRAPPER_H

#include "numpy_version.h"
#include "numpy/arrayobject.h"
#include "cwrapper.h"
#include "cwrapper_ndarrays.h"


#ifdef MOD_N1F9SNG7JW4B_WRAPPER

void bind_c_assemble_matrix_un_ex01(int64_t, int64_t, int64_t, int64_t, void*, int64_t, int64_t, void*, int64_t, int64_t, void*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, void*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, void*, int64_t, int64_t, int64_t, int64_t, void*, int64_t, int64_t, int64_t, int64_t, void*, int64_t, int64_t, int64_t, int64_t, void*, int64_t, int64_t, int64_t, int64_t, int64_t, double, double, void*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);

#else

static void** Pymod_n1f9sng7jw4b_API;


/*........................................*/
static int mod_n1f9sng7jw4b_import(void)
{
    PyObject* current_path;
    PyObject* stash_path;
    current_path = PySys_GetObject("path");
    stash_path = PyList_GetItem(current_path, 0);
    Py_INCREF(stash_path);
    PyList_SetItem(current_path, 0, PyUnicode_FromString("/home/rifqui/Desktop/domain_decomposition_mutipatch/Isogeometric_analysis_and_domain_decomposition/parallel_Schwarz_method_Robin_2D/__epyccel__"));
    Pymod_n1f9sng7jw4b_API = (void**)PyCapsule_Import("mod_n1f9sng7jw4b._C_API", 0);
    PyList_SetItem(current_path, 0, stash_path);
    return Pymod_n1f9sng7jw4b_API != NULL ? 0 : -1;
}
/*........................................*/

#endif
#endif // MOD_N1F9SNG7JW4B_WRAPPER_H
