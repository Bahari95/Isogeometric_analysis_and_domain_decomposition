"""
# plot_patches : This script demonstrates how to create and visualize a multipatch configuration using the simplines library.
# 1. Import the necessary libraries
# 2. Define the degree and refinement level for the spline space
# 3. Specify the geometry file and the list of indices for the patches
# 4. Initialize the spline spaces and geometry mappings for each patch
# 5. Plot the Jacobian and mesh for the multipatch configuration
#   @ M. Bahari : TODO 2.5d and 3d parallelization and post-processing using VTK
# """
from simplines import SplineSpace
from simplines import TensorSpace
from simplines import getGeometryMap

#...
from   simplines                    import plot_JacobianMultipatch
from   simplines                    import plot_MeshMultipatch

degree      = 2  # fixed by parameterization for now
quad_degree = degree + 1
NRefine     = 4 # nelements refined NRefine times 

#----------------------------------------
#..... Parameterization from 16*16 elements
#----------------------------------------
# Quart annulus
#geometry  = './fields/quart_annulus.xml'
# Half annulus
geometry  = './fields/annulus_48.xml'
ListINdex = [11, 21, 31, 41, 51, 61, 71, 81, 12, 22, 32, 42, 52, 62, 72, 82, 13, 23, 33, 43, 53, 63, 73, 83, 14, 24, 34, 44, 54, 64, 74, 84]
# Circle
#geometry = './fields/circle.xml'
# Lshape
#geometry  = './fields/lshape.xml'
#ListINdex = [0, 1]
# DDM shape
#geometry  = './fields/ddm.xml'
# DDM shape
# geometry  = './fields/ddm2.xml'
# DDM shape
#geometry  = './fields/ddm3.xml'
# ... Overlape ??
#geometry  = './fields/Annulus_over1.xml'

print('#--- Plot geometry : ', geometry)

nbpts       = 100 # number of points for plot

#--------------------------
#..... Initialisation
#--------------------------
# create the spline space for each direction
V   = []
xmp = []
ymp = []
for i in ListINdex:
    print('--- Patch : ', i)
    # ... Assembling mapping
    mp         = getGeometryMap(geometry,i)

    # ... Refine number of elements
    nelements   = (mp.nelements[0] * NRefine, mp.nelements[1] * NRefine) #... number of elements
    print('Number of elements in each direction : ', nelements)

    # ... Refine mapping
    xmp1, ymp1  =  mp.RefineGeometryMap(Nelements= nelements)
    xmp.append(xmp1)
    ymp.append(ymp1)
    # ... create the spline space for each direction
    V1_0        = SplineSpace(degree=mp.degree[0], nelements= nelements[0])
    V2_0        = SplineSpace(degree=mp.degree[1], nelements= nelements[1])
    V.append(TensorSpace(V1_0, V2_0))

plot_JacobianMultipatch(nbpts, V, xmp, ymp)
plot_MeshMultipatch(nbpts, V, xmp, ymp)
