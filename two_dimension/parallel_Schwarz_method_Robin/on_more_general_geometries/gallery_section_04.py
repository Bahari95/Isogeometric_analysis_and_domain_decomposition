__all__ = ['assemble_matrix_ex01',
           'assemble_vector_ex01',
           'assemble_norm_ex01',
           'assemble_matrix_ex02',
           'assemble_vector_ex02',
           'assemble_norm_ex02'
]

from pyccel.decorators import types

#==============================================================================
# .. in uniform mesh Matrix
@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'int', 'real', 'real', 'double[:,:,:,:]')
def assemble_matrix_un_ex01(ne1, ne2,
                        p1, p2,
                        spans_1, spans_2,
                        basis_1, basis_2,
                        weights_1, weights_2,
                        points_1, points_2,
                        domain_nb, S_DDM,
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

    #.. circle

    # ...
    lcoeffs_u  = zeros((p1+1,p2+1))
    arr_J_mat0 = zeros((k1,k2))
    arr_J_mat1 = zeros((k1,k2))
    arr_J_mat2 = zeros((k1,k2))
    arr_J_mat3 = zeros((k1,k2))
    J_mat      = zeros((k1,k2))

    # ... build matrices
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    x1       = points_1[ie1,g1]
                    x2       = points_2[ie2,g2]


                    F1x      = 2.0*sqrt(1.0-0.5*(2.0*x2-1.0)**2)
                    F1y      = -(2.0*x1-1.0)*(2.0*x2-1.0)/sqrt(1.0-0.5*(2.0*x2-1.0)**2)
                    F2y      = 2.0*sqrt(1.0-0.5*(2.0*x1-1.0)**2)
                    F2x      = -(2.0*x1-1.0)*(2.0*x2-1.0)/sqrt(1.0-0.5*(2.0*x1-1.0)**2)
                    det_Hess = abs(F1x*F2y-F1y*F2x)

                    #F=(F1, F2)
                    arr_J_mat0[g1,g2] = F2y
                    arr_J_mat1[g1,g2] = F1x
                    arr_J_mat2[g1,g2] = F1y
                    arr_J_mat3[g1,g2] = F2x

                    J_mat[g1,g2]      = det_Hess
                                        
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

                                    bi_x1 = basis_1[ie1, il_1, 1, g1] * basis_2[ie2, il_2, 0, g2]
                                    bi_x2 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 1, g2]

                                    bj_x1 = basis_1[ie1, jl_1, 1, g1] * basis_2[ie2, jl_2, 0, g2]
                                    bj_x2 = basis_1[ie1, jl_1, 0, g1] * basis_2[ie2, jl_2, 1, g2]

                                    bi_x = arr_J_mat0[g1,g2] * bi_x1 - arr_J_mat3[g1,g2] * bi_x2
                                    bi_y = arr_J_mat1[g1,g2] * bi_x2 - arr_J_mat2[g1,g2] * bi_x1

                                    bj_x = arr_J_mat0[g1,g2] * bj_x1 - arr_J_mat3[g1,g2] * bj_x2 
                                    bj_y = arr_J_mat1[g1,g2] * bj_x2 - arr_J_mat2[g1,g2] * bj_x1 


                                    wvol = weights_1[ie1, g1] * weights_2[ie2, g2]

                                    v   += (bi_x * bj_x + bi_y * bj_y ) * wvol / J_mat[g1,g2]

                            matrix[p1+i1, p2+i2, p1+j1-i1, p2+j2-i2]  += v
    # ... build matrices
    if domain_nb == 0:
         ie1      = ne1 -1
         i_span_1 = spans_1[ie1]
         for ie2 in range(0, ne2):
                i_span_2 = spans_2[ie2]

                for g2 in range(0, k2):

                    x1       = ovlp_value
                    x2       = points_2[ie2,g2]


                    F1x      = 2.0*sqrt(1.0-0.5*(2.0*x2-1.0)**2)
                    F1y      = -(2.0*x1-1.0)*(2.0*x2-1.0)/sqrt(1.0-0.5*(2.0*x2-1.0)**2)
                    F2y      = 2.0*sqrt(1.0-0.5*(2.0*x1-1.0)**2)
                    F2x      = -(2.0*x1-1.0)*(2.0*x2-1.0)/sqrt(1.0-0.5*(2.0*x1-1.0)**2)
                    det_Hess = sqrt(F2y**2+F1y**2)

                    J_mat[0, g2]      = det_Hess
                    
                for il_2 in range(0, p2+1):
                        for jl_2 in range(0, p2+1):

                            i2 = i_span_2 - p2 + il_2
                            j2 = i_span_2 - p2 + jl_2

                            v  = 0.0
                            for g2 in range(0, k2):
                                    bi_0  = basis_2[ie2, il_2, 0, g2]
                                    bj_0  = basis_2[ie2, jl_2, 0, g2]

                                    wvol  = weights_2[ie2, g2] * J_mat[0, g2]
                                    # ...
                                    v    += S_DDM * bi_0 * bj_0 * wvol

                            matrix[i_span_1+p1, p2+i2, p1, p2+j2-i2]  += v
    else :
         ie1  = 0
         i_span_1 = spans_1[ie1]
         for ie2 in range(0, ne2):
                i_span_2 = spans_2[ie2]
                for g2 in range(0, k2):

                    x1       = ovlp_value
                    x2       = points_2[ie2,g2]


                    F1x      = 2.0*sqrt(1.0-0.5*(2.0*x2-1.0)**2)
                    F1y      = -(2.0*x1-1.0)*(2.0*x2-1.0)/sqrt(1.0-0.5*(2.0*x2-1.0)**2)
                    F2y      = 2.0*sqrt(1.0-0.5*(2.0*x1-1.0)**2)
                    F2x      = -(2.0*x1-1.0)*(2.0*x2-1.0)/sqrt(1.0-0.5*(2.0*x1-1.0)**2)
                    det_Hess = sqrt(F2y**2+F1y**2)

                    J_mat[0, g2]      = det_Hess
                    
                for il_2 in range(0, p2+1):
                        for jl_2 in range(0, p2+1):

                            i2 = i_span_2 - p2 + il_2
                            j2 = i_span_2 - p2 + jl_2

                            v  = 0.0
                            for g2 in range(0, k2):
                                    bi_0  = basis_2[ie2, il_2, 0, g2]
                                    bj_0  = basis_2[ie2, jl_2, 0, g2]

                                    wvol  = weights_2[ie2, g2] * J_mat[0, g2]
                                    # ...
                                    # ...
                                    v    += S_DDM * bi_0 * bj_0 * wvol

                            matrix[p1, p2+i2, p1, p2+j2-i2]  += v
    # ...

#==============================================================================Assemble rhs Poisson
#---1 : In uniform mesh
@types('int', 'int', 'int', 'int','int', 'int', 'int', 'int', 'int[:]', 'int[:]','int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'real[:]', 'real[:]', 'real[:]', 'real[:]', 'double[:,:]', 'real', 'real', 'int', 'double[:,:]')
def assemble_vector_ex01(ne1, ne2, ne3, ne4, p1, p2, p3, p4, spans_1, spans_2,  spans_3, spans_4, basis_1, basis_2, basis_3, basis_4, weights_1, weights_2, weights_3, weights_4, points_1, points_2, points_3, points_4, knots_1, knots_2, knots_3, knots_4, vector_d, ovlp_value, S_DDM, domain_nb, rhs):

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
    lvalues_f   = zeros((k1, k2))
    # ...
    J_mat       = zeros((k1, k2))
    #J_mat0      = zeros((k1, k2))
    #J_mat1      = zeros((k1, k2))
    #J_mat2      = zeros((k1, k2))
    #J_mat3      = zeros((k1, k2))
    # ...
    lcoeffs_d   = zeros((p1+1,p2+1))
    # ... build rhs
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]
            
            for g1 in range(0, k1):
                for g2 in range(0, k2):
                    x1            = points_1[ie1,g1]
                    x2            = points_2[ie2,g2]

                    x             = (2.0*x1-1.0)*sqrt(1.0-0.5*(2.0*x2-1.0)**2)
                    y             = (2.0*x2-1.0)*sqrt(1.0-0.5*(2.0*x1-1.0)**2)
                    
                    F1x           = 2.0*sqrt(1.0-0.5*(2.0*x2-1.0)**2)
                    F1y           = -(2.0*x1-1.0)*(2.0*x2-1.0)/sqrt(1.0-0.5*(2.0*x2-1.0)**2)
                    F2y           = 2.0*sqrt(1.0-0.5*(2.0*x1-1.0)**2)
                    F2x           = -(2.0*x1-1.0)*(2.0*x2-1.0)/sqrt(1.0-0.5*(2.0*x1-1.0)**2)
                    J_mat[g1,g2]  = abs(F1x*F2y-F1y*F2x)
                    #..F=(F1, F2)
                    #J_mat0[g1,g2] = F2y
                    #J_mat1[g1,g2] = F1x
                    #J_mat2[g1,g2] = F1y
                    #J_mat3[g1,g2] = F2x
                    #.. Test 1
                    #f = -4000000*x**2*(x**2 + y**2 - 0.2)**2*exp(-500*(x**2 + y**2 - 0.2)**2) + 4000*x**2*exp(-500*(x**2 + y**2 - 0.2)**2) - 4000000*y**2*(x**2 + y**2 - 0.2)**2*exp(-500*(x**2 + y**2 - 0.2)**2)
                    #f+= 4000*y**2*exp(-500*(x**2 + y**2 - 0.2)**2) - 2*(-2000*x**2 - 2000*y**2 + 400.0)*exp(-500*(x**2 + y**2 - 0.2)**2)

                    #.. Test 2
                    f = 4.

                    lvalues_f[g1,g2]   = f 
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    i1 = i_span_1 - p1 + il_1
                    i2 = i_span_2 - p2 + il_2
                    v = 0.0
                    for g1 in range(0, k1):
                        for g2 in range(0, k2):
                            
                            bi_0 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]

                            wvol  = weights_1[ie1, g1]*weights_2[ie2, g2] * J_mat[g1,g2]

                            # .. 
                            u    = lvalues_f[g1,g2]
                            # ..
                            v += bi_0 * u * wvol

                    rhs[i1+p1,i2+p2] += v  

    lvalues_u      = zeros(k2)
    lvalues_ux     = zeros(k2)
    #---Computes All basis in a new points
    nders          = 1
    degree         = p3
    #..
    left           = empty( degree )
    right          = empty( degree )
    a              = empty( (       2, degree+1) )
    ndu            = empty( (degree+1, degree+1) )
    ders           = zeros( (     nders+1, degree+1) ) # output array
    #...
    basis3         = zeros((nders+1, degree+1))
    for i in range(1):
            #span = find_span( knots, degree, xq )
            xq = ovlp_value
            #~~~~~~~~~~~~~~~
            #span = find_span( knots, degree, xq )
            #~~~~~~~~~~~~~~~
            # Knot index at left/right boundary
            low  = degree
            high = len(knots_3)-1-degree
            # Check if point is exactly on left/right boundary, or outside domain
            if xq <= knots_3[low ]: 
                 span = low
            if xq >= knots_3[high]: 
                 span = high-1
            else :
              # Perform binary search
              span = (low+high)//2
              while xq < knots_3[span] or xq >= knots_3[span+1]:
                 if xq < knots_3[span]:
                     high = span
                 else:
                     low  = span
                 span = (low+high)//2
            ndu[0,0] = 1.0
            for j in range(0,degree):
                left [j] = xq - knots_3[span-j]
                right[j] = knots_3[span+1+j] - xq
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
            # Multiply derivatives by correct factors
            r = degree
            ders[1,:] = ders[1,:] * r
            basis3[0,:] = ders[0,:]
            basis3[1,:] = ders[1,:]
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Assembles Neumann Condition
    if domain_nb == 0 :
      i1_ovrlp  = ne1+2*p1-1
      neum_sign1 = -1.
      neum_sign2 = 1.
    else :
      i1_ovrlp  = p1
      neum_sign1 = 1.
      neum_sign2 = -1.
                
    for ie2 in range(0, ne2):
           i_span_2 = spans_2[ie2]
           
           for g2 in range(0, k2):
                    #... We compute firstly the span in new adapted points                             
                    #...                    
                    xq        = ovlp_value
                    degree    = p3
                    low       = degree
                    high      = len(knots_3)-1-degree
                    # Check if point is exactly on left/right boundary, or outside domain
                    if xq <= knots_3[low ]: 
                         span = low
                    if xq >= knots_3[high]: 
                         span = high-1
                    else :
                      # Perform binary search
                      span = (low+high)//2
                      while xq < knots_3[span] or xq >= knots_3[span+1]:
                         if xq < knots_3[span]:
                             high = span
                         else:
                             low  = span
                         span = (low+high)//2
                    span_3    = span 
                    # ...
                    lcoeffs_d[:, :] = vector_d[span_3 : span_3+p3+1, i_span_2 : i_span_2+p4+1]
                    # ...
                    u   = 0.
                    ux  = 0.
                    uy  = 0.
                    for il_1 in range(0, p3+1):
                       for il_2 in range(0, p4+1):
                             bi_0      = basis3[0,il_1] * basis_4[ie2, il_2, 0, g2]
                             bi_x      = basis3[1,il_1] * basis_4[ie2, il_2, 0, g2]
                             bi_y      = basis3[0,il_1] * basis_4[ie2, il_2, 1, g2]
                             # ...
                             coeff_d   = lcoeffs_d[il_1, il_2]
                             # ...
                             u        +=  coeff_d*bi_0
                             ux       +=  coeff_d*bi_x
                             uy       +=  coeff_d*bi_y
                    # ....
                    x1       = ovlp_value
                    x2       = points_2[ie2,g2]

                    F1    = (2.0*x1-1.0)*sqrt(1.0-0.5*(2.0*x2-1.0)**2)
                    F2    = (2.0*x2-1.0)*sqrt(1.0-0.5*(2.0*x1-1.0)**2)
                    F1x   = 2.0*sqrt(1.0-0.5*(2.0*x2-1.0)**2)
                    F1y   = -(2.0*x1-1.0)*(2.0*x2-1.0)/sqrt(1.0-0.5*(2.0*x2-1.0)**2)

                    F2y   = 2.0*sqrt(1.0-0.5*(2.0*x1-1.0)**2)
                    F2x   = -(2.0*x1-1.0)*(2.0*x2-1.0)/sqrt(1.0-0.5*(2.0*x1-1.0)**2)

                    det_Hess = abs(F1x*F2y-F1y*F2x)
                    # ...
                    #comp_1         = ( F2y*ux - F2x*uy)/det_Hess * tang_1/sqrt(tang_1**2+tang_2**2)
                    #comp_2         = (-F1y*ux + F1x*uy)/det_Hess * tang_2/sqrt(tang_1**2+tang_2**2)
                    # ...
                    comp_1         = neum_sign1 * ( F2y*ux - F2x*uy)/det_Hess * F2y #/sqrt(F1y**2+ F2y**2)
                    comp_2         = neum_sign2 * (-F1y*ux + F1x*uy)/det_Hess * F1y #/sqrt(F1y**2+ F2y**2)
                    # ...
                    lvalues_u[g2]  = u * sqrt(F1y**2+ F2y**2)
                    lvalues_ux[g2] = -(comp_1 + comp_2) #* sqrt(F1y**2+ F2y**2)
                                                
           for il_2 in range(0, p2+1):
                 i2 = i_span_2 - p2 + il_2

                 v = 0.0
                 for g2 in range(0, k2):
                       bi_0     =  basis_2[ie2, il_2, 0, g2]
                       wsurf    =  weights_2[ie2, g2]
                  
                       #.. 
                       v       += bi_0 * (lvalues_ux[g2]+S_DDM * lvalues_u[g2]) * wsurf
                      
                 rhs[i1_ovrlp,i2+p2] += v
    # ...


#=================================================================================
# norm in uniform mesh norm
@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]',  'double[:,:]',  'double[:,:]')
def assemble_norm_ex01(ne1, ne2, p1, p2, spans_1, spans_2,  basis_1, basis_2,  weights_1, weights_2, points_1, points_2, vector_u,rhs):

    from numpy import exp
    from numpy import pi
    from numpy import sin, sinh
    from numpy import arctan2
    from numpy import cos, cosh
    from numpy import sqrt
    from numpy import zeros
    # ... sizes
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]

    #.. circle

    # ...
    lcoeffs_u  = zeros((p1+1,p2+1))
    lvalues_ux  = zeros((k1, k2))
    lvalues_uy  = zeros((k1, k2))
    lvalues_u  = zeros((k1, k2))

    error_l2 = 0.
    error_H1 = 0.
    # ...
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            lvalues_u[ : , : ]  = 0.0
            lvalues_ux[ : , : ]  = 0.0
            lvalues_uy[ : , : ]  = 0.0
            lcoeffs_u[ : , : ] = vector_u[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    coeff_u = lcoeffs_u[il_1,il_2]

                    for g1 in range(0, k1):
                        b1 = basis_1[ie1,il_1,0,g1]
                        db1 = basis_1[ie1,il_1,1,g1]
                        for g2 in range(0, k2):
                            b2 = basis_2[ie2,il_2,0,g2]
                            db2 = basis_2[ie2,il_2,1,g2]

                            lvalues_u[g1,g2]   += coeff_u*b1*b2
                            lvalues_ux[g1,g2]  += coeff_u*db1*b2
                            lvalues_uy[g1,g2]  += coeff_u*b1*db2

            w = 0.0
            v = 0.0
            for g1 in range(0, k1):
                for g2 in range(0, k2):
                    wvol = weights_1[ie1, g1] * weights_2[ie2, g2]
                    x1    =  points_1[ie1, g1]
                    x2    =  points_2[ie2, g2]

                    uh  = lvalues_u[g1,g2]
                    sx  = lvalues_ux[g1,g2]
                    sy  = lvalues_uy[g1,g2]

                    x = (2.0*x1-1.0)*sqrt(1.0-0.5*(2.0*x2-1.0)**2)
                    y = (2.0*x2-1.0)*sqrt(1.0-0.5*(2.0*x1-1.0)**2)
                    
                    F1x   = 2.0*sqrt(1.0-0.5*(2.0*x2-1.0)**2)
                    F1y   = -(2.0*x1-1.0)*(2.0*x2-1.0)/sqrt(1.0-0.5*(2.0*x2-1.0)**2)
                    F2y   = 2.0*sqrt(1.0-0.5*(2.0*x1-1.0)**2)
                    F2x   = -(2.0*x1-1.0)*(2.0*x2-1.0)/sqrt(1.0-0.5*(2.0*x1-1.0)**2)
                    det_J = abs(F1x*F2y-F1y*F2x)
                    
                    #... TEST 1
                    #f   = exp(-500*(x**2 + y**2 - 0.2)**2)
                    #fx  = -2000*x*(x**2 + y**2 - 0.2)*exp(-500*(x**2 + y**2 - 0.2)**2)
                    #fy  = -2000*y*(x**2 + y**2 - 0.2)*exp(-500*(x**2 + y**2 - 0.2)**2)

                    #... TEST 2
                    f    = 1.-x**2-y**2
                    fx   = -2*x
                    fy   = -2*y
                    
                    uhx = (F2y*sx-F2x*sy)/det_J

                    uhy = (F1x*sy-F1y*sx)/det_J

                    w  += ((uhx-fx)**2 +(uhy-fy)**2)* wvol * det_J
                    v  += (uh-f)**2 * wvol * det_J

            error_H1      += w
            error_l2      += v
    rhs[p1,p2]   = sqrt(error_l2)
    rhs[p1,p2+1] = sqrt(error_H1)
    #...
