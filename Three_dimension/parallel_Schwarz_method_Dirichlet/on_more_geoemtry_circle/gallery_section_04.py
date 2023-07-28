__all__ = ['assemble_matrix_ex01',
           'assemble_vector_ex01',
           'assemble_norm_ex01',
           'assemble_matrix_ex02',
           'assemble_vector_ex02',
           'assemble_norm_ex02'
]

from pyccel.decorators import types

# assembles mass matrix 1D
#==============================================================================
@types('int', 'int', 'int[:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]')
def assemble_massmatrix1D(ne, degree, spans, basis, weights, points, matrix):

    # ... sizes
    k1 = weights.shape[1]
    # ... build matrices
    for ie1 in range(0, ne):
            i_span_1 = spans[ie1]        
            # evaluation dependant uniquement de l'element

            for il_1 in range(0, degree+1):
                i1 = i_span_1 - degree + il_1
                for il_2 in range(0, degree+1):
                            i2 = i_span_1 - degree + il_2
                            v = 0.0
                            for g1 in range(0, k1):
                                
                                    bi_0 = basis[ie1, il_1, 0, g1]
                                    bj_0 = basis[ie1, il_2, 0, g1]
                                    
                                    wvol = weights[ie1, g1]
                                    
                                    v += bi_0 * bj_0 * wvol

                            matrix[degree+i1, degree+ i2-i1]  += v
    # ...
    
#==============================================================================
@types('int', 'int', 'int[:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]')
def assemble_matrix_ex11(ne, degree, spans, basis, weights, points,  matrix):

    # ... sizes
    k1 = weights.shape[1]
    # ... build matrices
    for ie1 in range(0, ne):
            i_span_1 = spans[ie1]        
            # evaluation dependant uniquement de l'element

            for il_1 in range(0, degree+1):
                i1 = i_span_1 - degree + il_1
                for il_2 in range(0, degree+1):
                            i2 = i_span_1 - degree + il_2
                            v = 0.0
                            for g1 in range(0, k1):
                                
                                    bi_x = basis[ie1, il_1, 1, g1]
                                    bj_x = basis[ie1, il_2, 0, g1]
                                    
                                    wvol = weights[ie1, g1]
                                    
                                    v += bi_x * bj_x * wvol

                            matrix[ degree+ i1, degree+ i2-i1]  += v
    # ...
#==============================================================================
@types('int', 'int', 'int[:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]')
def assemble_matrix_ex12(ne, degree, spans, basis, weights, points,  matrix):
    # ... sizes
    k1 = weights.shape[1]
    # ... build matrices
    for ie1 in range(0, ne):
            i_span_1 = spans[ie1]        
            # evaluation dependant uniquement de l'element

            for il_1 in range(0, degree+1):
                i1 = i_span_1 - degree + il_1
                for il_2 in range(0, degree+1):
                            i2 = i_span_1 - degree + il_2
                            v = 0.0
                            for g1 in range(0, k1):
                                
                                    bi_x = basis[ie1, il_1, 0, g1]
                                    bj_x = basis[ie1, il_2, 1, g1]
                                    
                                    wvol = weights[ie1, g1]
                                    
                                    v += bi_x * bj_x * wvol

                            matrix[ degree+ i1, degree+ i2-i1]  += v
    # ...    
# assembles stiffness matrix 1D
#==============================================================================
@types('int', 'int', 'int[:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]')
def assemble_stiffnessmatrix1D(ne, degree, spans, basis, weights, points,  matrix):

    # ... sizes
    k1 = weights.shape[1]
    # ... build matrices
    for ie1 in range(0, ne):
            i_span_1 = spans[ie1]        
            # evaluation dependant uniquement de l'element

            for il_1 in range(0, degree+1):
                i1 = i_span_1 - degree + il_1
                for il_2 in range(0, degree+1):
                            i2 = i_span_1 - degree + il_2
                            v = 0.0
                            for g1 in range(0, k1):
                                
                                    bi_x = basis[ie1, il_1, 1, g1]
                                    bj_x = basis[ie1, il_2, 1, g1]
                                    
                                    wvol = weights[ie1, g1]
                                    
                                    v += bi_x * bj_x * wvol

                            matrix[ degree+ i1, degree+ i2-i1]  += v

#==============================================================================
# .. in uniform mesh Matrix
@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'real', 'double[:,:,:,:]')
def assemble_matrix_un_ex01(ne1, ne2,
                        p1, p2,
                        spans_1, spans_2,
                        basis_1, basis_2,
                        weights_1, weights_2,
                        points_1, points_2,
                        ovlp_value, matrix):

    # ... sizes
    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import sqrt
    from numpy import zeros

    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]

    # ...
    lcoeffs_u  = zeros((p1+1,p2+1))
    J_mat      = zeros((k1,k2))
    # ... build matrices
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            for g1 in range(0, k1):
                 for g2 in range(0, k2):

                         x1    =  ovlp_value
                         x2    =  points_1[ie1, g1]
                         x3    =  points_2[ie2, g2]

                         F1   = (2.0*x1-1.0) * sqrt(1.-0.5*(2.0*x2-1.0)**2-0.5*(2.0*x3-1.0)**2+(2.0*x2-1.0)**2*(2.0*x3-1.0)**2/3)
                         F1x  = 2.0 * sqrt(1.-0.5*(2.0*x2-1.0)**2-0.5*(2.0*x3-1.0)**2+(2.0*x2-1.0)**2*(2.0*x3-1.0)**2/3)
                         F1y  = (2.0*x1-1.0) *(-(2.0*x2-1.0)+(2.0*x2-1.0)*(2.0*x3-1.0)**2*2/3)
                         F1y /= sqrt(1.-0.5*(2.0*x2-1.0)**2-0.5*(2.0*x3-1.0)**2+(2.0*x2-1.0)**2*(2.0*x3-1.0)**2/3)
                         F1z  = (2.0*x1-1.0) * (-(2.0*x3-1.0)+(2.0*x2-1.0)**2*(2.0*x3-1.0)*2/3)
                         F1z /= sqrt(1.-0.5*(2.0*x2-1.0)**2-0.5*(2.0*x3-1.0)**2+(2.0*x2-1.0)**2*(2.0*x3-1.0)**2/3)
 
                         F2   = (2.0*x2-1.0) * sqrt(1.-0.5*(2.0*x1-1.0)**2-0.5*(2.0*x3-1.0)**2+(2.0*x1-1.0)**2*(2.0*x3-1.0)**2/3)
                         F2y  = 2.0 * sqrt(1.-0.5*(2.0*x1-1.0)**2-0.5*(2.0*x3-1.0)**2+(2.0*x1-1.0)**2*(2.0*x3-1.0)**2/3)
                         F2x  = (2.0*x2-1.0) *(-(2.0*x1-1.0)+(2.0*x1-1.0)*(2.0*x3-1.0)**2*2/3)
                         F2x /= sqrt(1.-0.5*(2.0*x1-1.0)**2-0.5*(2.0*x3-1.0)**2+(2.0*x1-1.0)**2*(2.0*x3-1.0)**2/3)
                         F2z  = (2.0*x2-1.0) * (-(2.0*x3-1.0)+(2.0*x1-1.0)**2*(2.0*x3-1.0)*2/3)
                         F2z /= sqrt(1.-0.5*(2.0*x1-1.0)**2-0.5*(2.0*x3-1.0)**2+(2.0*x1-1.0)**2*(2.0*x3-1.0)**2/3)

                         F3   = (2.0*x3-1.0) * sqrt(1.-0.5*(2.0*x1-1.0)**2-0.5*(2.0*x2-1.0)**2+(2.0*x1-1.0)**2*(2.0*x2-1.0)**2/3)
                         F3z  = 2.0 * sqrt(1.-0.5*(2.0*x1-1.0)**2-0.5*(2.0*x2-1.0)**2+(2.0*x1-1.0)**2*(2.0*x2-1.0)**2/3)
                         F3x  = (2.0*x3-1.0) *(-(2.0*x1-1.0)+(2.0*x1-1.0)*(2.0*x2-1.0)**2*2/3)
                         F3x /= sqrt(1.-0.5*(2.0*x1-1.0)**2-0.5*(2.0*x2-1.0)**2+(2.0*x1-1.0)**2*(2.0*x2-1.0)**2/3)
                         F3y  = (2.0*x3-1.0) * (-(2.0*x2-1.0)+(2.0*x1-1.0)**2*(2.0*x2-1.0)*2/3)
                         F3y /= sqrt(1.-0.5*(2.0*x1-1.0)**2-0.5*(2.0*x2-1.0)**2+(2.0*x1-1.0)**2*(2.0*x2-1.0)**2/3)

                         det  = abs(F1x*(F2y*F3z-F3y*F2z)-F2x*(F1y*F3z-F3y*F1z)+F3x*(F1y*F2z-F2y*F1z) )
                         J_mat[g1,g2]       = det
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    for jl_1 in range(0, p1+1):
                        for jl_2 in range(0, p2+1):

                            i1 = i_span_1 - p1 + il_1
                            j1 = i_span_1 - p1 + jl_1

                            i2 = i_span_2 - p2 + il_2
                            j2 = i_span_2 - p2 + jl_2

                            v  = 0.0
                            for g1 in range(0, k1):
                                for g2 in range(0, k2):

                                    bi_0  = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]
                                    bj_0  = basis_1[ie1, jl_1, 0, g1] * basis_2[ie2, jl_2, 0, g2]
                                    # ...
                                    wvol = weights_1[ie1, g1] * weights_2[ie2, g2] * J_mat[g1,g2]
                                    # ...
                                    v += bj_0* bi_0 * wvol

                            matrix[p1+i1, p2+i2, p1+j1-i1, p2+j2-i2]  += v
    # ...


#==============================================================================Assembles stiffness matrix
#---1 : In uniform mesh
@types('int', 'int', 'int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:,:,:,:,:]')
def assemble_matrix_ex01(ne1, ne2, ne3,
                        p1, p2, p3,
                        spans_1, spans_2, spans_3, 
                        basis_1, basis_2, basis_3,
                        weights_1, weights_2, weights_3,
                        points_1, points_2, points_3,
                        matrix):
    
    from numpy import exp
    from numpy import pi
    from numpy import sin, sinh
    from numpy import cos, cosh
    from numpy import arctan2
    from numpy import sqrt
    from numpy import zeros
    from numpy import empty

    # ... Sphere

    # ... sizes
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    k3 = weights_3.shape[1]
    # ...

    # ...
    lcoeffs_u  = zeros((p1+1,p2+1,p3+1))

    # ... Co matrix coeffs
    arr_J_mat11 = zeros((k1,k2,k3))
    arr_J_mat12 = zeros((k1,k2,k3))
    arr_J_mat13 = zeros((k1,k2,k3))

    arr_J_mat21 = zeros((k1,k2,k3))
    arr_J_mat22 = zeros((k1,k2,k3))
    arr_J_mat23 = zeros((k1,k2,k3))

    arr_J_mat31 = zeros((k1,k2,k3))
    arr_J_mat32 = zeros((k1,k2,k3))
    arr_J_mat33 = zeros((k1,k2,k3))

    J_mat      = zeros((k1,k2,k3))

    # ... build matrices
    for ie1 in range(0, ne1):
      i_span_1 = spans_1[ie1]
      for ie2 in range(0, ne2):
        i_span_2 = spans_2[ie2]
        for ie3 in range(0, ne3):
           i_span_3 = spans_3[ie3]

           for g1 in range(0, k1):
              for g2 in range(0, k2):
                 for g3 in range(0, k3):

                         x1    =  points_1[ie1, g1]
                         x2    =  points_2[ie2, g2]
                         x3    =  points_3[ie3, g3]

                         F1   = (2.0*x1-1.0) * sqrt(1.-0.5*(2.0*x2-1.0)**2-0.5*(2.0*x3-1.0)**2+(2.0*x2-1.0)**2*(2.0*x3-1.0)**2/3)
                         F1x  = 2.0 * sqrt(1.-0.5*(2.0*x2-1.0)**2-0.5*(2.0*x3-1.0)**2+(2.0*x2-1.0)**2*(2.0*x3-1.0)**2/3)
                         F1y  = (2.0*x1-1.0) *(-(2.0*x2-1.0)+(2.0*x2-1.0)*(2.0*x3-1.0)**2*2/3)
                         F1y /= sqrt(1.-0.5*(2.0*x2-1.0)**2-0.5*(2.0*x3-1.0)**2+(2.0*x2-1.0)**2*(2.0*x3-1.0)**2/3)
                         F1z  = (2.0*x1-1.0) * (-(2.0*x3-1.0)+(2.0*x2-1.0)**2*(2.0*x3-1.0)*2/3)
                         F1z /= sqrt(1.-0.5*(2.0*x2-1.0)**2-0.5*(2.0*x3-1.0)**2+(2.0*x2-1.0)**2*(2.0*x3-1.0)**2/3)
 
                         F2   = (2.0*x2-1.0) * sqrt(1.-0.5*(2.0*x1-1.0)**2-0.5*(2.0*x3-1.0)**2+(2.0*x1-1.0)**2*(2.0*x3-1.0)**2/3)
                         F2y  = 2.0 * sqrt(1.-0.5*(2.0*x1-1.0)**2-0.5*(2.0*x3-1.0)**2+(2.0*x1-1.0)**2*(2.0*x3-1.0)**2/3)
                         F2x  = (2.0*x2-1.0) *(-(2.0*x1-1.0)+(2.0*x1-1.0)*(2.0*x3-1.0)**2*2/3)
                         F2x /= sqrt(1.-0.5*(2.0*x1-1.0)**2-0.5*(2.0*x3-1.0)**2+(2.0*x1-1.0)**2*(2.0*x3-1.0)**2/3)
                         F2z  = (2.0*x2-1.0) * (-(2.0*x3-1.0)+(2.0*x1-1.0)**2*(2.0*x3-1.0)*2/3)
                         F2z /= sqrt(1.-0.5*(2.0*x1-1.0)**2-0.5*(2.0*x3-1.0)**2+(2.0*x1-1.0)**2*(2.0*x3-1.0)**2/3)

                         F3   = (2.0*x3-1.0) * sqrt(1.-0.5*(2.0*x1-1.0)**2-0.5*(2.0*x2-1.0)**2+(2.0*x1-1.0)**2*(2.0*x2-1.0)**2/3)
                         F3z  = 2.0 * sqrt(1.-0.5*(2.0*x1-1.0)**2-0.5*(2.0*x2-1.0)**2+(2.0*x1-1.0)**2*(2.0*x2-1.0)**2/3)
                         F3x  = (2.0*x3-1.0) *(-(2.0*x1-1.0)+(2.0*x1-1.0)*(2.0*x2-1.0)**2*2/3)
                         F3x /= sqrt(1.-0.5*(2.0*x1-1.0)**2-0.5*(2.0*x2-1.0)**2+(2.0*x1-1.0)**2*(2.0*x2-1.0)**2/3)
                         F3y  = (2.0*x3-1.0) * (-(2.0*x2-1.0)+(2.0*x1-1.0)**2*(2.0*x2-1.0)*2/3)
                         F3y /= sqrt(1.-0.5*(2.0*x1-1.0)**2-0.5*(2.0*x2-1.0)**2+(2.0*x1-1.0)**2*(2.0*x2-1.0)**2/3)

                         det  = abs(F1x*(F2y*F3z-F3y*F2z)-F2x*(F1y*F3z-F3y*F1z)+F3x*(F1y*F2z-F2y*F1z) )

                         arr_J_mat11[g1,g2,g3] = F2y*F3z-F2z*F3y
                         arr_J_mat12[g1,g2,g3] = F2z*F3x-F2x*F3z
                         arr_J_mat13[g1,g2,g3] = F2x*F3y-F2y*F3x

                         arr_J_mat21[g1,g2,g3] = F1z*F3y-F1y*F3z
                         arr_J_mat22[g1,g2,g3] = F1x*F3z-F3x*F1z
                         arr_J_mat23[g1,g2,g3] = F1y*F3x-F1x*F3y

                         arr_J_mat31[g1,g2,g3] = F1y*F2z-F2y*F1z
                         arr_J_mat32[g1,g2,g3] = F2x*F1z-F1x*F2z
                         arr_J_mat33[g1,g2,g3] = F1x*F2y-F2x*F1y

                         J_mat[g1,g2,g3]       = det

           for il_1 in range(0, p1+1):
             for il_2 in range(0, p2+1):
                for il_3 in range(0, p3+1):

                   for jl_1 in range(0, p1+1):  
                      for jl_2 in range(0, p2+1):
                         for jl_3 in range(0, p3+1):
                            i1 = i_span_1 - p1 + il_1
                            j1 = i_span_1 - p1 + jl_1
        
                            i2 = i_span_2 - p2 + il_2
                            j2 = i_span_2 - p2 + jl_2

                            i3 = i_span_3 - p3 + il_3
                            j3 = i_span_3 - p3 + jl_3

                            v = 0.0
                            for g1 in range(0, k1):
                               for g2 in range(0, k2):
                                  for g3 in range(0, k3):

                                     bi_x1 = basis_1[ie1, il_1, 1, g1] * basis_2[ie2, il_2, 0, g2] * basis_3[ie3, il_3, 0, g3]
                                     bi_x2 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 1, g2] * basis_3[ie3, il_3, 0, g3]
                                     bi_x3 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2] * basis_3[ie3, il_3, 1, g3]

                                     bj_x1 = basis_1[ie1, jl_1, 1, g1] * basis_2[ie2, jl_2, 0, g2] * basis_3[ie3, jl_3, 0, g3]
                                     bj_x2 = basis_1[ie1, jl_1, 0, g1] * basis_2[ie2, jl_2, 1, g2] * basis_3[ie3, jl_3, 0, g3]
                                     bj_x3 = basis_1[ie1, jl_1, 0, g1] * basis_2[ie2, jl_2, 0, g2] * basis_3[ie3, jl_3, 1, g3]


                                     bi_x = arr_J_mat11[g1,g2,g3] * bi_x1 + arr_J_mat12[g1,g2,g3] * bi_x2 + arr_J_mat13[g1,g2,g3] * bi_x3
                                     bi_y = arr_J_mat21[g1,g2,g3] * bi_x1 + arr_J_mat22[g1,g2,g3] * bi_x2 + arr_J_mat23[g1,g2,g3] * bi_x3
                                     bi_z = arr_J_mat31[g1,g2,g3] * bi_x1 + arr_J_mat32[g1,g2,g3] * bi_x2 + arr_J_mat33[g1,g2,g3] * bi_x3

                                     bj_x = arr_J_mat11[g1,g2,g3] * bj_x1 + arr_J_mat12[g1,g2,g3] * bj_x2 + arr_J_mat13[g1,g2,g3] * bj_x3
                                     bj_y = arr_J_mat21[g1,g2,g3] * bj_x1 + arr_J_mat22[g1,g2,g3] * bj_x2 + arr_J_mat23[g1,g2,g3] * bj_x3
                                     bj_z = arr_J_mat31[g1,g2,g3] * bj_x1 + arr_J_mat32[g1,g2,g3] * bj_x2 + arr_J_mat33[g1,g2,g3] * bj_x3

                                     wvol = weights_1[ie1, g1] * weights_2[ie2, g2] * weights_3[ie3, g3] / J_mat[g1,g2,g3]

                                     v += (bi_x * bj_x + bi_y * bj_y + bi_z * bj_z ) * wvol

                            matrix[p1+i1, p2+i2, p3+i3, p1+j1-i1, p2+j2-i2, p3+j3-i3]  += v
    # ...
    	
#==============================================================================Assemble rhs Poisson
#---1 : In uniform mesh
@types('int', 'int', 'int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:,:]', 'double[:,:,:]')
def assemble_vector_ex01(ne1, ne2, ne3, p1, p2, p3, spans_1, spans_2, spans_3,  basis_1, basis_2, basis_3,  weights_1, weights_2, weights_3, points_1, points_2, points_3, vector_d, rhs):

    from numpy import exp
    from numpy import pi
    from numpy import sin
    from numpy import arctan2
    from numpy import cos, cosh
    from numpy import sqrt
    from numpy import zeros

    # ... sizes
    k1        = weights_1.shape[1]
    k2        = weights_2.shape[1]
    k3        = weights_3.shape[1]

    # ...
    lcoeffs_d   = zeros((p1+1,p2+1,p3+1))
    # ..
    lvalues_u   = zeros((k1, k2, k3))
    lvalues_udx = zeros((k1, k2, k3))
    lvalues_udy = zeros((k1, k2, k3))
    lvalues_udz = zeros((k1, k2, k3))
    # ... Co matrix coeffs
    arr_J_mat11 = zeros((k1,k2,k3))
    arr_J_mat12 = zeros((k1,k2,k3))
    arr_J_mat13 = zeros((k1,k2,k3))

    arr_J_mat21 = zeros((k1,k2,k3))
    arr_J_mat22 = zeros((k1,k2,k3))
    arr_J_mat23 = zeros((k1,k2,k3))

    arr_J_mat31 = zeros((k1,k2,k3))
    arr_J_mat32 = zeros((k1,k2,k3))
    arr_J_mat33 = zeros((k1,k2,k3))
    # ...
    J_mat      = zeros((k1,k2,k3))        
    # ... build rhs
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]
            for ie3 in range(0, ne3):
                i_span_3 = spans_3[ie3]

                lcoeffs_d[ : , : , : ] = vector_d[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1, i_span_3 : i_span_3+p3+1]
                for g1 in range(0, k1):
                    for g2 in range(0, k2):
                        for g3 in range(0, k3):
                            
                            ux = 0.
                            uy = 0.
                            uz = 0.
                            for il_1 in range(0, p1+1):
                              for il_2 in range(0, p2+1):
                                for il_3 in range(0, p3+1):
                                    
                                    bi_x = basis_1[ie1, il_1, 1, g1] * basis_2[ie2, il_2, 0, g2] * basis_3[ie3, il_3, 0, g3]
                                    bi_y = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 1, g2] * basis_3[ie3, il_3, 0, g3]                              
                                    bi_z = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2] * basis_3[ie3, il_3, 1, g3]
   
                                    coeff_d = lcoeffs_d[il_1, il_2, il_3]

                                    ux     +=  coeff_d*bi_x
                                    uy     +=  coeff_d*bi_y
                                    uz     +=  coeff_d*bi_z
                    
                            x1    =  points_1[ie1, g1]
                            x2    =  points_2[ie2, g2]
                            x3    =  points_3[ie3, g3]

                            F1   = (2.0*x1-1.0) * sqrt(1.-0.5*(2.0*x2-1.0)**2-0.5*(2.0*x3-1.0)**2+(2.0*x2-1.0)**2*(2.0*x3-1.0)**2/3)
                            F1x  = 2.0 * sqrt(1.-0.5*(2.0*x2-1.0)**2-0.5*(2.0*x3-1.0)**2+(2.0*x2-1.0)**2*(2.0*x3-1.0)**2/3)
                            F1y  = (2.0*x1-1.0) *(-(2.0*x2-1.0)+(2.0*x2-1.0)*(2.0*x3-1.0)**2*2/3)
                            F1y /= sqrt(1.-0.5*(2.0*x2-1.0)**2-0.5*(2.0*x3-1.0)**2+(2.0*x2-1.0)**2*(2.0*x3-1.0)**2/3)
                            F1z  = (2.0*x1-1.0) * (-(2.0*x3-1.0)+(2.0*x2-1.0)**2*(2.0*x3-1.0)*2/3)
                            F1z /= sqrt(1.-0.5*(2.0*x2-1.0)**2-0.5*(2.0*x3-1.0)**2+(2.0*x2-1.0)**2*(2.0*x3-1.0)**2/3)
 
                            F2   = (2.0*x2-1.0) * sqrt(1.-0.5*(2.0*x1-1.0)**2-0.5*(2.0*x3-1.0)**2+(2.0*x1-1.0)**2*(2.0*x3-1.0)**2/3)
                            F2y  = 2.0 * sqrt(1.-0.5*(2.0*x1-1.0)**2-0.5*(2.0*x3-1.0)**2+(2.0*x1-1.0)**2*(2.0*x3-1.0)**2/3)
                            F2x  = (2.0*x2-1.0) *(-(2.0*x1-1.0)+(2.0*x1-1.0)*(2.0*x3-1.0)**2*2/3)
                            F2x /= sqrt(1.-0.5*(2.0*x1-1.0)**2-0.5*(2.0*x3-1.0)**2+(2.0*x1-1.0)**2*(2.0*x3-1.0)**2/3)
                            F2z  = (2.0*x2-1.0) * (-(2.0*x3-1.0)+(2.0*x1-1.0)**2*(2.0*x3-1.0)*2/3)
                            F2z /= sqrt(1.-0.5*(2.0*x1-1.0)**2-0.5*(2.0*x3-1.0)**2+(2.0*x1-1.0)**2*(2.0*x3-1.0)**2/3)

                            F3   = (2.0*x3-1.0) * sqrt(1.-0.5*(2.0*x1-1.0)**2-0.5*(2.0*x2-1.0)**2+(2.0*x1-1.0)**2*(2.0*x2-1.0)**2/3)
                            F3z  = 2.0 * sqrt(1.-0.5*(2.0*x1-1.0)**2-0.5*(2.0*x2-1.0)**2+(2.0*x1-1.0)**2*(2.0*x2-1.0)**2/3)
                            F3x  = (2.0*x3-1.0) *(-(2.0*x1-1.0)+(2.0*x1-1.0)*(2.0*x2-1.0)**2*2/3)
                            F3x /= sqrt(1.-0.5*(2.0*x1-1.0)**2-0.5*(2.0*x2-1.0)**2+(2.0*x1-1.0)**2*(2.0*x2-1.0)**2/3)
                            F3y  = (2.0*x3-1.0) * (-(2.0*x2-1.0)+(2.0*x1-1.0)**2*(2.0*x2-1.0)*2/3)
                            F3y /= sqrt(1.-0.5*(2.0*x1-1.0)**2-0.5*(2.0*x2-1.0)**2+(2.0*x1-1.0)**2*(2.0*x2-1.0)**2/3) 
                            # ...
                            arr_J_mat11[g1,g2,g3] = F2y*F3z-F2z*F3y
                            arr_J_mat12[g1,g2,g3] = F2z*F3x-F2x*F3z
                            arr_J_mat13[g1,g2,g3] = F2x*F3y-F2y*F3x

                            arr_J_mat21[g1,g2,g3] = F1z*F3y-F1y*F3z
                            arr_J_mat22[g1,g2,g3] = F1x*F3z-F3x*F1z
                            arr_J_mat23[g1,g2,g3] = F1y*F3x-F1x*F3y

                            arr_J_mat31[g1,g2,g3] = F1y*F2z-F2y*F1z
                            arr_J_mat32[g1,g2,g3] = F2x*F1z-F1x*F2z
                            arr_J_mat33[g1,g2,g3] = F1x*F2y-F2x*F1y
                            #....
                            lvalues_udx[g1, g2, g3] = arr_J_mat11[g1,g2,g3] * ux + arr_J_mat12[g1,g2,g3] * uy + arr_J_mat13[g1,g2,g3] * uz
                            lvalues_udy[g1, g2, g3] = arr_J_mat21[g1,g2,g3] * ux + arr_J_mat22[g1,g2,g3] * uy + arr_J_mat23[g1,g2,g3] * uz
                            lvalues_udz[g1, g2, g3] = arr_J_mat31[g1,g2,g3] * ux + arr_J_mat32[g1,g2,g3] * uy + arr_J_mat33[g1,g2,g3] * uz
                            # ...
                            x = F1
                            y = F2
                            z = F3
                            det = abs(F1x*(F2y*F3z-F3y*F2z)-F2x*(F1y*F3z-F3y*F1z)+F3x*(F1y*F2z-F2y*F1z) )
                            # ... Test 0
                            f = 6.
                            # ... Test 2
                            #f  = 4*x*(-50.0*sinh(50*x + 50*y + 50*z - 12.5)/cosh(50*x + 50*y + 50*z - 12.5)**2 - 50.0*sinh(50*x + 50*y + 50*z + 12.5)/cosh(50*x + 50*y + 50*z + 12.5)**2) 
                            #f += + 4*y*(-50.0*sinh(50*x + 50*y + 50*z - 12.5)/cosh(50*x + 50*y + 50*z - 12.5)**2 - 50.0*sinh(50*x + 50*y + 50*z + 12.5)/cosh(50*x + 50*y + 50*z + 12.5)**2) 
                            #f += 4*z*(-50.0*sinh(50*x + 50*y + 50*z - 12.5)/cosh(50*x + 50*y + 50*z - 12.5)**2 - 50.0*sinh(50*x + 50*y + 50*z + 12.5)/cosh(50*x + 50*y + 50*z + 12.5)**2)
                            #f += - 3*(-x**2 - y**2 - z**2 + 1.0)*(5000.0*sinh(50*x + 50*y + 50*z - 12.5)**2/cosh(50*x + 50*y + 50*z - 12.5)**3 + 5000.0*sinh(50*x + 50*y + 50*z + 12.5)**2/cosh(50*x + 50*y + 50*z + 12.5)**3 - 2500.0/cosh(50*x + 50*y + 50*z + 12.5) - 2500.0/cosh(50*x + 50*y + 50*z - 12.5)) 
                            #f += 6.0 + 6.0/cosh(50*x + 50*y + 50*z + 12.5) + 6.0/cosh(50*x + 50*y + 50*z - 12.5)

                            # ... Test 3
                            #f = -4000000*x**2*(x**2 + y**2 + z**2 - 0.2)**2*exp(-500*(x**2 + y**2 + z**2 - 0.2)**2) + 4000*x**2*exp(-500*(x**2 + y**2 + z**2 - 0.2)**2) 
                            #f+= - 4000000*y**2*(x**2 + y**2 + z**2 - 0.2)**2*exp(-500*(x**2 + y**2 + z**2 - 0.2)**2) + 4000*y**2*exp(-500*(x**2 + y**2 + z**2 - 0.2)**2) 
                            #f+= - 4000000*z**2*(x**2 + y**2 + z**2 - 0.2)**2*exp(-500*(x**2 + y**2 + z**2 - 0.2)**2) + 4000*z**2*exp(-500*(x**2 + y**2 + z**2 - 0.2)**2) - 3*(-2000*x**2 - 2000*y**2 - 2000*z**2 + 400.0)*exp(-500*(x**2 + y**2 + z**2 - 0.2)**2)
                            
                            lvalues_u[g1,g2,g3]  = f
                            J_mat[g1,g2,g3]      = det

                for il_1 in range(0, p1+1):
                  for il_2 in range(0, p2+1):
                    for il_3 in range(0, p3+1):

                      i1 = i_span_1 - p1 + il_1
                      i2 = i_span_2 - p2 + il_2
                      i3 = i_span_3 - p3 + il_3

                      v = 0.0
                      for g1 in range(0, k1):
                         for g2 in range(0, k2):
                            for g3 in range(0, k3):

                              bi_0  = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2] * basis_3[ie3, il_3, 0, g3]
                              bi_x1 = basis_1[ie1, il_1, 1, g1] * basis_2[ie2, il_2, 0, g2] * basis_3[ie3, il_3, 0, g3]
                              bi_x2 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 1, g2] * basis_3[ie3, il_3, 0, g3]                              
                              bi_x3 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2] * basis_3[ie3, il_3, 1, g3]
                              # ...
                              bi_x  = arr_J_mat11[g1,g2,g3] * bi_x1 + arr_J_mat12[g1,g2,g3] * bi_x2 + arr_J_mat13[g1,g2,g3] * bi_x3
                              bi_y  = arr_J_mat21[g1,g2,g3] * bi_x1 + arr_J_mat22[g1,g2,g3] * bi_x2 + arr_J_mat23[g1,g2,g3] * bi_x3
                              bi_z  = arr_J_mat31[g1,g2,g3] * bi_x1 + arr_J_mat32[g1,g2,g3] * bi_x2 + arr_J_mat33[g1,g2,g3] * bi_x3
                              #..
                              wvol  = weights_1[ie1, g1] * weights_2[ie2, g2] * weights_3[ie3, g3]
                              #++
                              
                              ux    = lvalues_udx[g1,g2, g3]
                              uy    = lvalues_udy[g1,g2, g3]
                              uz    = lvalues_udz[g1,g2, g3]
                              #..
                              u     = lvalues_u[g1, g2, g3]
                                                                                          
                              v    += bi_0 * u * wvol * J_mat[g1,g2,g3] - (ux * bi_x + uy * bi_y + uz * bi_z) * wvol / J_mat[g1,g2,g3]

                      rhs[i1+p1,i2+p2,i3+p3] += v   
    # ...

#==============================================================================Assemble l2 and H1 error norm
#---1 : In uniform mesh
#==============================================================================
@types('int', 'int', 'int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:,:]', 'double[:,:,:]')
def assemble_norm_ex01(ne1, ne2, ne3, p1, p2, p3, spans_1, spans_2, spans_3,  basis_1, basis_2, basis_3,  weights_1, weights_2, weights_3, points_1, points_2, points_3, vector_u, rhs):

    from numpy import exp
    from numpy import pi
    from numpy import sin, sinh
    from numpy import cos, cosh
    from numpy import arctan2
    from numpy import sqrt
    from numpy import zeros
    from numpy import empty

    # ... Sphere

    # ... sizes
    k1         = weights_1.shape[1]
    k2         = weights_2.shape[1]
    k3         = weights_3.shape[1]

    # ...
    lcoeffs_u  = zeros((p1+1,p2+1,p3+1))

    # ...
    lvalues_u  = zeros((k1, k2, k3))
    lvalues_ux = zeros((k1, k2, k3))
    lvalues_uy = zeros((k1, k2, k3))
    lvalues_uz = zeros((k1, k2, k3))
    # ...

    # ...
    norm_H1    = 0.
    norm_l2    = 0.
    for ie1 in range(0, ne1):
       i_span_1 = spans_1[ie1]
       for ie2 in range(0, ne2):
          i_span_2 = spans_2[ie2]
          for ie3 in range(0, ne3):
             i_span_3 = spans_3[ie3]

             lvalues_u[ : , : , :]  = 0.0
             lvalues_ux[ : , : , :] = 0.0
             lvalues_uy[ : , : , :] = 0.0
             lvalues_uz[ : , : , :] = 0.0

             lcoeffs_u[ : , : , :]  = vector_u[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1, i_span_3 : i_span_3+p3+1]

             for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                   for il_3 in range(0, p3+1):
                      coeff_u = lcoeffs_u[il_1,il_2,il_3]

                      for g1 in range(0, k1):
                        b1  = basis_1[ie1,il_1,0,g1]
                        db1 = basis_1[ie1,il_1,1,g1]
                        for g2 in range(0, k2):
                            b2  = basis_2[ie2,il_2,0,g2]
                            db2 = basis_2[ie2,il_2,1,g2]
                            for g3 in range(0, k3):
                               b3  = basis_3[ie3,il_3,0,g3]
                               db3 = basis_3[ie3,il_3,1,g3]

                               lvalues_u[g1,g2,g3]  += coeff_u*b1*b2*b3
                               lvalues_ux[g1,g2,g3] += coeff_u*db1*b2*b3
                               lvalues_uy[g1,g2,g3] += coeff_u*b1*db2*b3
                               lvalues_uz[g1,g2,g3] += coeff_u*b1*b2*db3

             v = 0.0
             w = 0.0
             for g1 in range(0, k1):
               for g2 in range(0, k2):
                 for g3 in range(0, k3):

                         x1    = points_1[ie1, g1]
                         x2    = points_2[ie2, g2]
                         x3    = points_3[ie3, g3]

                         F1   = (2.0*x1-1.0) * sqrt(1.-0.5*(2.0*x2-1.0)**2-0.5*(2.0*x3-1.0)**2+(2.0*x2-1.0)**2*(2.0*x3-1.0)**2/3)
                         F1x  = 2.0 * sqrt(1.-0.5*(2.0*x2-1.0)**2-0.5*(2.0*x3-1.0)**2+(2.0*x2-1.0)**2*(2.0*x3-1.0)**2/3)
                         F1y  = (2.0*x1-1.0) *(-(2.0*x2-1.0)+(2.0*x2-1.0)*(2.0*x3-1.0)**2*2/3)
                         F1y /= sqrt(1.-0.5*(2.0*x2-1.0)**2-0.5*(2.0*x3-1.0)**2+(2.0*x2-1.0)**2*(2.0*x3-1.0)**2/3)
                         F1z  = (2.0*x1-1.0) * (-(2.0*x3-1.0)+(2.0*x2-1.0)**2*(2.0*x3-1.0)*2/3)
                         F1z /= sqrt(1.-0.5*(2.0*x2-1.0)**2-0.5*(2.0*x3-1.0)**2+(2.0*x2-1.0)**2*(2.0*x3-1.0)**2/3)
 
                         F2   = (2.0*x2-1.0) * sqrt(1.-0.5*(2.0*x1-1.0)**2-0.5*(2.0*x3-1.0)**2+(2.0*x1-1.0)**2*(2.0*x3-1.0)**2/3)
                         F2y  = 2.0 * sqrt(1.-0.5*(2.0*x1-1.0)**2-0.5*(2.0*x3-1.0)**2+(2.0*x1-1.0)**2*(2.0*x3-1.0)**2/3)
                         F2x  = (2.0*x2-1.0) *(-(2.0*x1-1.0)+(2.0*x1-1.0)*(2.0*x3-1.0)**2*2/3)
                         F2x /= sqrt(1.-0.5*(2.0*x1-1.0)**2-0.5*(2.0*x3-1.0)**2+(2.0*x1-1.0)**2*(2.0*x3-1.0)**2/3)
                         F2z  = (2.0*x2-1.0) * (-(2.0*x3-1.0)+(2.0*x1-1.0)**2*(2.0*x3-1.0)*2/3)
                         F2z /= sqrt(1.-0.5*(2.0*x1-1.0)**2-0.5*(2.0*x3-1.0)**2+(2.0*x1-1.0)**2*(2.0*x3-1.0)**2/3)

                         F3   = (2.0*x3-1.0) * sqrt(1.-0.5*(2.0*x1-1.0)**2-0.5*(2.0*x2-1.0)**2+(2.0*x1-1.0)**2*(2.0*x2-1.0)**2/3)
                         F3z  = 2.0 * sqrt(1.-0.5*(2.0*x1-1.0)**2-0.5*(2.0*x2-1.0)**2+(2.0*x1-1.0)**2*(2.0*x2-1.0)**2/3)
                         F3x  = (2.0*x3-1.0) *(-(2.0*x1-1.0)+(2.0*x1-1.0)*(2.0*x2-1.0)**2*2/3)
                         F3x /= sqrt(1.-0.5*(2.0*x1-1.0)**2-0.5*(2.0*x2-1.0)**2+(2.0*x1-1.0)**2*(2.0*x2-1.0)**2/3)
                         F3y  = (2.0*x3-1.0) * (-(2.0*x2-1.0)+(2.0*x1-1.0)**2*(2.0*x2-1.0)*2/3)
                         F3y /= sqrt(1.-0.5*(2.0*x1-1.0)**2-0.5*(2.0*x2-1.0)**2+(2.0*x1-1.0)**2*(2.0*x2-1.0)**2/3)

                         J_mat = abs(F1x*(F2y*F3z-F3y*F2z)-F2x*(F1y*F3z-F3y*F1z)+F3x*(F1y*F2z-F2y*F1z) )

                         mat11 = F2y*F3z-F2z*F3y
                         mat12 = F2z*F3x-F2x*F3z
                         mat13 = F2x*F3y-F2y*F3x

                         mat21 = F1z*F3y-F1y*F3z
                         mat22 = F1x*F3z-F3x*F1z
                         mat23 = F1y*F3x-F1x*F3y     

                         mat31 = F1y*F2z-F2y*F1z
                         mat32 = F2x*F1z-F1x*F2z
                         mat33 = F1x*F2y-F2x*F1y

                         # ...
                         x    =  F1
                         y    =  F2
                         z    =  F3

                         # ... TEST 3
                         u   =  1.-(x**2 + y**2 + z**2)
                         ux  = -2*x
                         uy  = -2*y
                         uz  = -2*z
                         
                         # ... TEST 2
                         #u   = (1. + 1./cosh(50 * ( x + y + z + 0.25  )) + 1.0/cosh(50 * ( x + y + z -0.25)))*(1.-x**2-y**2-z**2)
                         #ux  = -2*x*(1.0 + 1.0/cosh(50*x + 50*y + 50*z + 12.5) + 1.0/cosh(50*x + 50*y + 50*z - 12.5)) + (-50.0*sinh(50*x + 50*y + 50*z - 12.5)/cosh(50*x + 50*y + 50*z - 12.5)**2 - 50.0*sinh(50*x + 50*y + 50*z + 12.5)/cosh(50*x + 50*y + 50*z + 12.5)**2)*(-x**2 - y**2 - z**2 + 1.0)
                         #uy  = -2*y*(1.0 + 1.0/cosh(50*x + 50*y + 50*z + 12.5) + 1.0/cosh(50*x + 50*y + 50*z - 12.5)) + (-50.0*sinh(50*x + 50*y + 50*z - 12.5)/cosh(50*x + 50*y + 50*z - 12.5)**2 - 50.0*sinh(50*x + 50*y + 50*z + 12.5)/cosh(50*x + 50*y + 50*z + 12.5)**2)*(-x**2 - y**2 - z**2 + 1.0)
                         #uz  = -2*z*(1.0 + 1.0/cosh(50*x + 50*y + 50*z + 12.5) + 1.0/cosh(50*x + 50*y + 50*z - 12.5)) + (-50.0*sinh(50*x + 50*y + 50*z - 12.5)/cosh(50*x + 50*y + 50*z - 12.5)**2 - 50.0*sinh(50*x + 50*y + 50*z + 12.5)/cosh(50*x + 50*y + 50*z + 12.5)**2)*(-x**2 - y**2 - z**2 + 1.0)

                         # ... TEST 3
                         #u   = exp(-500*(x**2 + y**2 + z**2 - 0.2)**2)
                         #ux  = -2000*x*(x**2 + y**2 + z**2 - 0.2)*exp(-500*(x**2 + y**2 + z**2 - 0.2)**2)
                         #uy  = -2000*y*(x**2 + y**2 + z**2 - 0.2)*exp(-500*(x**2 + y**2 + z**2 - 0.2)**2)
                         #uz  = -2000*z*(x**2 + y**2 + z**2 - 0.2)*exp(-500*(x**2 + y**2 + z**2 - 0.2)**2)

                         uh   = lvalues_u[g1,g2,g3]
                         uhx1 = lvalues_ux[g1,g2,g3]
                         uhx2 = lvalues_uy[g1,g2,g3]
                         uhx3 = lvalues_uz[g1,g2,g3]

                         uhx  = (mat11 * uhx1 + mat12 * uhx2 + mat13 * uhx3) / J_mat
                         uhy  = (mat21 * uhx1 + mat22 * uhx2 + mat23 * uhx3) / J_mat
                         uhz  = (mat31 * uhx1 + mat32 * uhx2 + mat33 * uhx3) / J_mat
     
                         wvol = weights_1[ie1, g1] * weights_2[ie2, g2] * weights_3[ie3, g3]
     
                         v   += ((ux-uhx)**2+(uy-uhy)**2+(uz-uhz)**2) * wvol * J_mat
                         w   += (u-uh)**2 * wvol * J_mat

             norm_H1 += v
             norm_l2 += w

    norm_H1 = sqrt(norm_H1)
    norm_l2 = sqrt(norm_l2)

    rhs[p1,p2,p3]   = norm_l2
    rhs[p1,p2,p3+1] = norm_H1
    # ...
    
#==============================================================================Assemble rhs Poisson
#---1 : In uniform mesh
@types('int', 'int','int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'int[:]','int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]','double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'real[:]', 'real[:]', 'real[:]', 'real[:]', 'real[:]', 'double[:,:,:]', 'real', 'double[:,:]')
def assemble_vector_ex02(ne1, ne2, ne3, ne4, ne5, p1, p2, p3, p4, p5, spans_1, spans_2,  spans_3, spans_4, spans_5, basis_1, basis_2, basis_3, basis_4, basis_5, weights_1, weights_2, weights_3, weights_4, weights_5, points_1, points_2, points_3, points_4, points_5, knots_1, knots_2, knots_3, knots_4, knots_5, vector_d, ovlp_value, rhs):

    from numpy import exp
    from numpy import pi
    from numpy import sin
    from numpy import arctan2
    from numpy import cos, cosh
    from numpy import sqrt
    from numpy import zeros
    from numpy import empty

    # ... sizes
    k1        = weights_1.shape[1]
    k2        = weights_2.shape[1]
    # ...
    # ...
    lcoeffs_d   = zeros((p5+1,p3+1,p4+1))
    # ..
    lvalues_u   = zeros((k1, k2))
    #---Computes All basis in a new points
    nders          = 0
    degree         = p5
    #..
    left           = empty( degree )
    right          = empty( degree )
    a              = empty( (       2, degree+1) )
    ndu            = empty( (degree+1, degree+1) )
    ders           = zeros( (     nders+1, degree+1) ) # output array
    #...
    basis5         = zeros(degree+1)
    for i in range(1):
            #span = find_span( knots, degree, xq )
            xq = ovlp_value
            #~~~~~~~~~~~~~~~
            # Knot index at left/right boundary
            low  = degree
            high = len(knots_5)-1-degree
            # Check if point is exactly on left/right boundary, or outside domain
            if xq <= knots_5[low ]: 
                 span = low
            if xq >= knots_5[high]: 
                 span = high-1
            else :
              # Perform binary search
              span = (low+high)//2
              while xq < knots_5[span] or xq >= knots_5[span+1]:
                 if xq < knots_5[span]:
                     high = span
                 else:
                     low  = span
                 span = (low+high)//2
            ndu[0,0] = 1.0
            for j in range(0,degree):
                left [j] = xq - knots_5[span-j]
                right[j] = knots_5[span+1+j] - xq
                saved    = 0.0
                for r in range(0,j+1):
                    # compute inverse of knot differences and save them into lower triangular part of ndu
                    ndu[j+1,r] = 1.0 / (right[r] + left[j-r])
                    # compute basis functions and save them into upper triangular part of ndu
                    temp       = ndu[r,j] * ndu[j+1,r]
                    ndu[r,j+1] = saved + right[r] * temp
                    saved      = left[j-r] * temp
                ndu[j+1,j+1] = saved	

            # Compute derivatives in 2D output array 'ders'
            ders[0,:] = ndu[:,degree]
            for r in range(0,degree+1):
                s1 = 0
                s2 = 1
                a[0,0] = 1.0
                for k in range(1,nders+1):
                    d  = 0.0
                    rk = r-k
                    pk = degree-k
                    if r >= k:
                       a[s2,0] = a[s1,0] * ndu[pk+1,rk]
                       d = a[s2,0] * ndu[rk,pk]
                    j1 = 1   if (rk  > -1 ) else -rk
                    j2 = k-1 if (r-1 <= pk) else degree-r
                    for ij in range(j1,j2+1):
                        a[s2,ij] = (a[s1,ij] - a[s1,ij-1]) * ndu[pk+1,rk+ij]
                    for ij in range(j1,j2+1):
                        d += a[s2,ij]* ndu[rk+ij,pk]
                    if r <= pk:
                       a[s2,k] = - a[s1,k-1] * ndu[pk+1,r]
                       d += a[s2,k] * ndu[r,pk]
                    ders[k,r] = d
                    j  = s1
                    s1 = s2
                    s2 = j
            basis5[:] = ders[0,:]
    # ... build rhs
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            for g1 in range(0, k1):
                for g2 in range(0, k2):
                    x1    =  ovlp_value
                    x2    =  points_1[ie1, g1]
                    x3    =  points_2[ie2, g2]

                    F1   = (2.0*x1-1.0) * sqrt(1.-0.5*(2.0*x2-1.0)**2-0.5*(2.0*x3-1.0)**2+(2.0*x2-1.0)**2*(2.0*x3-1.0)**2/3)
                    F1x  = 2.0 * sqrt(1.-0.5*(2.0*x2-1.0)**2-0.5*(2.0*x3-1.0)**2+(2.0*x2-1.0)**2*(2.0*x3-1.0)**2/3)
                    F1y  = (2.0*x1-1.0) *(-(2.0*x2-1.0)+(2.0*x2-1.0)*(2.0*x3-1.0)**2*2/3)
                    F1y /= sqrt(1.-0.5*(2.0*x2-1.0)**2-0.5*(2.0*x3-1.0)**2+(2.0*x2-1.0)**2*(2.0*x3-1.0)**2/3)
                    F1z  = (2.0*x1-1.0) * (-(2.0*x3-1.0)+(2.0*x2-1.0)**2*(2.0*x3-1.0)*2/3)
                    F1z /= sqrt(1.-0.5*(2.0*x2-1.0)**2-0.5*(2.0*x3-1.0)**2+(2.0*x2-1.0)**2*(2.0*x3-1.0)**2/3)
 
                    F2   = (2.0*x2-1.0) * sqrt(1.-0.5*(2.0*x1-1.0)**2-0.5*(2.0*x3-1.0)**2+(2.0*x1-1.0)**2*(2.0*x3-1.0)**2/3)
                    F2y  = 2.0 * sqrt(1.-0.5*(2.0*x1-1.0)**2-0.5*(2.0*x3-1.0)**2+(2.0*x1-1.0)**2*(2.0*x3-1.0)**2/3)
                    F2x  = (2.0*x2-1.0) *(-(2.0*x1-1.0)+(2.0*x1-1.0)*(2.0*x3-1.0)**2*2/3)
                    F2x /= sqrt(1.-0.5*(2.0*x1-1.0)**2-0.5*(2.0*x3-1.0)**2+(2.0*x1-1.0)**2*(2.0*x3-1.0)**2/3)
                    F2z  = (2.0*x2-1.0) * (-(2.0*x3-1.0)+(2.0*x1-1.0)**2*(2.0*x3-1.0)*2/3)
                    F2z /= sqrt(1.-0.5*(2.0*x1-1.0)**2-0.5*(2.0*x3-1.0)**2+(2.0*x1-1.0)**2*(2.0*x3-1.0)**2/3)

                    F3   = (2.0*x3-1.0) * sqrt(1.-0.5*(2.0*x1-1.0)**2-0.5*(2.0*x2-1.0)**2+(2.0*x1-1.0)**2*(2.0*x2-1.0)**2/3)
                    F3z  = 2.0 * sqrt(1.-0.5*(2.0*x1-1.0)**2-0.5*(2.0*x2-1.0)**2+(2.0*x1-1.0)**2*(2.0*x2-1.0)**2/3)
                    F3x  = (2.0*x3-1.0) *(-(2.0*x1-1.0)+(2.0*x1-1.0)*(2.0*x2-1.0)**2*2/3)
                    F3x /= sqrt(1.-0.5*(2.0*x1-1.0)**2-0.5*(2.0*x2-1.0)**2+(2.0*x1-1.0)**2*(2.0*x2-1.0)**2/3)
                    F3y  = (2.0*x3-1.0) * (-(2.0*x2-1.0)+(2.0*x1-1.0)**2*(2.0*x2-1.0)*2/3)
                    F3y /= sqrt(1.-0.5*(2.0*x1-1.0)**2-0.5*(2.0*x2-1.0)**2+(2.0*x1-1.0)**2*(2.0*x2-1.0)**2/3)
                    det  = abs(F1x*(F2y*F3z-F3y*F2z)-F2x*(F1y*F3z-F3y*F1z)+F3x*(F1y*F2z-F2y*F1z) )
                    
                    #... We compute firstly the span in new adapted points                             
                    #...                    
                    xq        = ovlp_value
                    degree    = p5
                    low       = degree
                    high      = len(knots_5)-1-degree
                    # Check if point is exactly on left/right boundary, or outside domain
                    if xq <= knots_5[low ]: 
                         span = low
                    if xq >= knots_5[high]: 
                         span = high-1
                    else :
                      # Perform binary search
                      span = (low+high)//2
                      while xq < knots_5[span] or xq >= knots_5[span+1]:
                         if xq < knots_5[span]:
                             high = span
                         else:
                             low  = span
                         span = (low+high)//2
                    span_5    = span 
                    # ...
                    lcoeffs_d[:, : , :] = vector_d[span_5 : span_5+p5+1, i_span_1 : i_span_1+p3+1, i_span_2 : i_span_2+p4+1]
                    # ...
                    u  = 0.
                    for il_3 in range(0, p5+1):
                       for il_1 in range(0, p3+1):
                          for il_2 in range(0, p4+1):
                             bi_0    = basis5[il_3] * basis_3[ie1, il_1, 0, g1] * basis_4[ie2, il_2, 0, g2]
                             # ...
                             coeff_d = lcoeffs_d[il_3, il_1, il_2]
                             # ...
                             u      +=  coeff_d*bi_0
                    lvalues_u[g1, g2] = u * det

            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):

                      i1 = i_span_1 - p1 + il_1
                      i2 = i_span_2 - p2 + il_2

                      v = 0.0
                      for g1 in range(0, k1):
                         for g2 in range(0, k2):

                              bi_0 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]
                              # ...
                              wvol = weights_1[ie1, g1] * weights_2[ie2, g2]
                              # ...
                              u    = lvalues_u[g1, g2]
                              # ...        
                              v   += bi_0 * u * wvol

                      rhs[i1+p1,i2+p2] += v   
    # ...    
    
