#ifndef MOD_XDCVVYXDGZKT_WRAPPER_H
#define MOD_XDCVVYXDGZKT_WRAPPER_H

#include "numpy_version.h"
#include "numpy/arrayobject.h"
#include "cwrapper.h"
#include "cwrapper_ndarrays.h"


#ifdef MOD_XDCVVYXDGZKT_WRAPPER

void bind_c_assemble_norm_ex01(int64_t, int64_t, int64_t, int64_t, void*, int64_t, int64_t, void*, int64_t, int64_t, void*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, void*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, void*, int64_t, int64_t, int64_t, int64_t, void*, int64_t, int64_t, int64_t, int64_t, void*, int64_t, int64_t, int64_t, int64_t, void*, int64_t, int64_t, int64_t, int64_t, void*, int64_t, int64_t, int64_t, int64_t, void*, int64_t, int64_t, int64_t, int64_t);

#else

static void** Pymod_xdcvvyxdgzkt_API;


/*........................................*/
static int mod_xdcvvyxdgzkt_import(void)
{
    PyObject* current_path;
    PyObject* stash_path;
    current_path = PySys_GetObject("path");
    stash_path = PyList_GetItem(current_path, 0);
    Py_INCREF(stash_path);
    PyList_SetItem(current_path, 0, PyUnicode_FromString("/home/rifqui/Desktop/domain_decomposition_mutipatch/Isogeometric_analysis_and_domain_decomposition/parallel_Schwarz_method_Robin_2D/__epyccel__"));
    Pymod_xdcvvyxdgzkt_API = (void**)PyCapsule_Import("mod_xdcvvyxdgzkt._C_API", 0);
    PyList_SetItem(current_path, 0, stash_path);
    return Pymod_xdcvvyxdgzkt_API != NULL ? 0 : -1;
}
/*........................................*/

#endif
#endif // MOD_XDCVVYXDGZKT_WRAPPER_H
