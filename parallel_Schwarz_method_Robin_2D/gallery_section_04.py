__all__ = ['assemble_matrix_un_ex01',
           'assemble_vector_un_ex01',
           'assemble_norm_ex01'
]

from pyccel.decorators import types
@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]','double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'int', 'float', 'float', 'double[:,:,:,:]')
def assemble_matrix_un_ex01(ne1, ne2,
                        p1, p2,
                        spans_1, spans_2,
                        basis_1, basis_2,
                        weights_1, weights_2,
                        points_1, points_2,
                        vector_m1, vector_m2,
                        domain_nb, S_DDM,
                        ovlp_value, matrix):	
    from numpy import zeros
    from numpy import sqrt
    # ...
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
   
    lcoeffs_m1 = zeros((p1+1,p2+1))
    lcoeffs_m2 = zeros((p1+1,p2+1))

    # ...
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
            
            lcoeffs_m1[ : , : ] = vector_m1[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_m2[ : , : ] = vector_m2[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    F1x = 0.0
                    F1y = 0.0
                    F2x = 0.0
                    F2y = 0.0
                    for il_1 in range(0, p1+1):
                          for il_2 in range(0, p2+1):

                              bj_x     = basis_1[ie1,il_1,1,g1]*basis_2[ie2,il_2,0,g2]
                              bj_y     = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,1,g2]

                              coeff_m1 = lcoeffs_m1[il_1,il_2]
                              F1x     +=  coeff_m1 * bj_x
                              F1y     +=  coeff_m1 * bj_y

                              coeff_m2 = lcoeffs_m2[il_1,il_2]
                              F2x     +=  coeff_m2 * bj_x
                              F2y     +=  coeff_m2 * bj_y

                    # ...
                    arr_J_mat0[g1,g2] = F2y
                    arr_J_mat1[g1,g2] = F1x
                    arr_J_mat2[g1,g2] = F1y
                    arr_J_mat3[g1,g2] = F2x

                    J_mat[g1,g2] = abs(F1x*F2y-F1y*F2x)

            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    for jl_1 in range(0, p1+1):
                        for jl_2 in range(0, p2+1):

                            i1 = i_span_1 - p1 + il_1
                            j1 = i_span_1 - p1 + jl_1

                            i2 = i_span_2 - p2 + il_2
                            j2 = i_span_2 - p2 + jl_2

                            v = 0.0
                            for g1 in range(0, k1):
                                for g2 in range(0, k2):
                                    bi_0  = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]
                                    bj_0  = basis_1[ie1, jl_1, 0, g1] * basis_2[ie2, jl_2, 0, g2]

                                    bi_x1 = basis_1[ie1, il_1, 1, g1] * basis_2[ie2, il_2, 0, g2]
                                    bi_x2 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 1, g2]

                                    bj_x1 = basis_1[ie1, jl_1, 1, g1] * basis_2[ie2, jl_2, 0, g2]
                                    bj_x2 = basis_1[ie1, jl_1, 0, g1] * basis_2[ie2, jl_2, 1, g2]

                                    bi_x  = arr_J_mat0[g1,g2] * bi_x1 - arr_J_mat3[g1,g2] * bi_x2
                                    bi_y  = arr_J_mat1[g1,g2] * bi_x2 - arr_J_mat2[g1,g2] * bi_x1

                                    bj_x  = arr_J_mat0[g1,g2] * bj_x1 - arr_J_mat3[g1,g2] * bj_x2 
                                    bj_y  = arr_J_mat1[g1,g2] * bj_x2 - arr_J_mat2[g1,g2] * bj_x1 


                                    wvol  = weights_1[ie1, g1] * weights_2[ie2, g2]

                                    v    += (bi_x * bj_x + bi_y * bj_y ) * wvol / J_mat[g1,g2]

                            matrix[p1+i1, p2+i2, p1+j1-i1, p2+j2-i2]  += v
   
    if domain_nb == 0:
        #... Assemble the boundary condition for Roben (x=right)
        ie1      = ne1 -1
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):         
            i_span_2 = spans_2[ie2]
            
            lcoeffs_m1[ : , : ] = vector_m1[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_m2[ : , : ] = vector_m2[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g2 in range(0, k2):

                F1y = 0.0
                F2y = 0.0
                for il_2 in range(0, p2+1):

                    bj_y     = basis_2[ie2,il_2,1,g2]

                    coeff_m1 = lcoeffs_m1[p1, il_2]
                    coeff_m2 = lcoeffs_m2[p1, il_2]                    

                    F1y     +=  coeff_m1 * bj_y
                    F2y     +=  coeff_m2 * bj_y

                J_mat[0 ,g2] = sqrt(F1y**2 + F2y**2)

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
        #... Assemble the boundary condition for Roben (x=left)
        ie1      = 0
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            lcoeffs_m1[ : , : ] = vector_m1[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_m2[ : , : ] = vector_m2[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g2 in range(0, k2):

                F1y = 0.0
                F2y = 0.0
                for il_2 in range(0, p2+1):

                    bj_y     = basis_2[ie2,il_2,1,g2]
                    coeff_m1 = lcoeffs_m1[0, il_2]
                    coeff_m2 = lcoeffs_m2[0, il_2]

                    F1y     +=  coeff_m1 * bj_y
                    F2y     +=  coeff_m2 * bj_y

                J_mat[0 ,g2] = sqrt(F1y**2 + F2y**2)
            for il_2 in range(0, p2+1):
                for jl_2 in range(0, p2+1):
                    i2 = i_span_2 - p2 + il_2
                    j2 = i_span_2 - p2 + jl_2
                    v  = 0.0
                    for g2 in range(0, k2):
                        bi_0  = basis_2[ie2, il_2, 0, g2]
                        bj_0  = basis_2[ie2, jl_2, 0, g2]
                        wvol  = weights_2[ie2, g2] * J_mat[0, g2]

                        v    += S_DDM * bi_0 * bj_0 * wvol

                    matrix[p1, p2+i2, p1, p2+j2-i2]  += v
    # ...


#==============================================================================Assemble rhs Poisson
##---2 : rhs
@types(	'int', 'int','int', 'int', 'int', 'int', 'int', 'int', 'int[:]',  'int[:]',  'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'float[:]', 'float[:]', 'float[:]', 'float[:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'int', 'float', 'float', 'double[:,:]')
def assemble_vector_un_ex01(ne1, ne2, ne3, ne4,
			    	p1, p2, p3, p4, 
    			    spans_1, spans_2, spans_3, spans_4,
    			    basis_1, basis_2, basis_3, basis_4,
    			    weights_1, weights_2, weights_3, weights_4,
    			    points_1, points_2, points_3, points_4,
    			    knots_1, knots_2, knots_3, knots_4,
    			    vector_m1, vector_m2, vector_d,
                    vector_m3, vector_m4, vector_np,
    			    domain_nb, S_DDM, ovlp_value, rhs):

    from numpy import exp
    from numpy import empty
    from numpy import cos, cosh
    from numpy import sin, sinh
    from numpy import pi
    from numpy import arctan2
    from numpy import sqrt
    from numpy import zeros

    # ... sizes
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]

    # ...
    lcoeffs_m1  = zeros((p1+1,p2+1))
    lcoeffs_m2  = zeros((p1+1,p2+1))
    lcoeffs_di  = zeros((p1+1,p2+1))
    #..
    lvalues_u   = zeros((k1, k2))
    arr_J_mat0  = zeros((k1, k2))
    arr_J_mat1  = zeros((k1, k2))
    arr_J_mat2  = zeros((k1, k2))
    arr_J_mat3  = zeros((k1, k2))
    lvalues_Jac = zeros((k1, k2))
    lvalues_udx = zeros((k1, k2))
    lvalues_udy = zeros((k1, k2))

    # ... build rhs
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]
            
            lcoeffs_m1[ : , : ] = vector_m1[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_m2[ : , : ] = vector_m2[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_di[ : , : ] = vector_d[ i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    x    = 0.0
                    y    = 0.0
                    F1x  = 0.0
                    F1y  = 0.0
                    F2x  = 0.0
                    F2y  = 0.0
                    # ...
                    udx  = 0.0
                    udy  = 0.0
                    for il_1 in range(0, p1+1):
                          for il_2 in range(0, p2+1):

                              bj_0     = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,0,g2]
                              bj_x     = basis_1[ie1,il_1,1,g1]*basis_2[ie2,il_2,0,g2]
                              bj_y     = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,1,g2]

                              coeff_m1 =  lcoeffs_m1[il_1,il_2]
                              x       +=  coeff_m1 * bj_0
                              F1x     +=  coeff_m1 * bj_x
                              F1y     +=  coeff_m1 * bj_y

                              coeff_m2 =  lcoeffs_m2[il_1,il_2]
                              y       +=  coeff_m2 * bj_0
                              F2x     +=  coeff_m2 * bj_x
                              F2y     +=  coeff_m2 * bj_y

                              coeff_di =  lcoeffs_di[il_1,il_2]
                              udx     +=  coeff_di * bj_x
                              udy     +=  coeff_di * bj_y
                              
                    J_mat = abs(F1x*F2y-F1y*F2x)
                    arr_J_mat0[g1,g2] = F2y
                    arr_J_mat1[g1,g2] = F1x
                    arr_J_mat2[g1,g2] = F1y
                    arr_J_mat3[g1,g2] = F2x
                    lvalues_udx[g1, g2]  = (F2y * udx - F2x*udy)
                    lvalues_udx[g1, g2] /= J_mat
                    lvalues_udy[g1, g2]  = (F1x * udy - F1y*udx)
                    lvalues_udy[g1, g2] /= J_mat
                    #.. Test 1
                    f = 2.*(2.*pi)**2*sin(2.*pi*x)*sin(2.*pi*y)

                    lvalues_u[g1,g2]   = f 
                    lvalues_Jac[g1,g2] = J_mat
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    i1 = i_span_1 - p1 + il_1
                    i2 = i_span_2 - p2 + il_2

                    v = 0.0
                    for g1 in range(0, k1):
                        for g2 in range(0, k2):
                            bi_0  = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]
                            bi_x1 = basis_1[ie1, il_1, 1, g1] * basis_2[ie2, il_2, 0, g2]
                            bi_x2 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 1, g2]
                            # ...
                            wvol  = weights_1[ie1, g1] * weights_2[ie2, g2]
                            # ...
                            bi_x  = arr_J_mat0[g1,g2] * bi_x1 - arr_J_mat3[g1,g2] * bi_x2
                            bi_y  = arr_J_mat1[g1,g2] * bi_x2 - arr_J_mat2[g1,g2] * bi_x1
                            
                            u     = lvalues_u[g1,g2]
                            udx   = lvalues_udx[g1, g2]
                            udy   = lvalues_udy[g1, g2]
                            v    += bi_0 * u * wvol * lvalues_Jac[g1,g2] -  (udx * bi_x+ udy * bi_y) * wvol

                    rhs[i1+p1,i2+p2] += v
    lvalues_u2     = zeros(k2)
    lvalues_ux     = zeros(k2)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    # Assembles Neumann Condition
    if domain_nb == 0 :
        # ...
        ie1       = 0
        i_span_3  = spans_3[ie1]
        for ie2 in range(0, ne2):
           i_span_2 = spans_2[ie2]
           
           for g2 in range(0, k2):               
                    #... here we use the information from the second domain
                    lcoeffs_m1[ : , : ] = vector_m3[i_span_3 : i_span_3+p3+1, i_span_2 : i_span_2+p2+1]
                    lcoeffs_m2[ : , : ] = vector_m4[i_span_3 : i_span_3+p3+1, i_span_2 : i_span_2+p2+1]
                    #... correspond to the solution of the second domain
                    lcoeffs_di[ : , : ] = vector_np[i_span_3 : i_span_3+p3+1, i_span_2 : i_span_2+p2+1]
                    # ...
                    u    = 0.
                    ux   = 0.
                    uy   = 0.
                    F1y  = 0.
                    F2y  = 0.
                    F1x  = 0.
                    F2x  = 0.
                    for jl_2 in range(0, p4+1):
                        bi_0       = basis_4[ie2, jl_2, 0, g2]
                        bi_y       = basis_4[ie2, jl_2, 1, g2]
                        # ...
                        coeff_m1   = lcoeffs_m1[0, jl_2]
                        coeff_m2   = lcoeffs_m2[0, jl_2]
                        coeff_m3   = lcoeffs_m1[1, jl_2]
                        coeff_m4   = lcoeffs_m2[1, jl_2]

                        F1y       +=  coeff_m1 * bj_y
                        F2y       +=  coeff_m2 * bj_y
                        F1x       +=  (coeff_m3-coeff_m1) * bj_0 * p3 / knots_3[p3+1]
                        F2x       +=  (coeff_m4-coeff_m2) * bj_0 * p3 / knots_3[p3+1]

                        coeff_d   = lcoeffs_di[0, jl_2]
                        coeff_d2  = lcoeffs_di[1, jl_2]
                        # ...
                        u        +=  coeff_d*bi_0
                        ux       +=  (coeff_d2-coeff_d)*bi_0 * p3 / knots_3[p3+1]
                        uy       +=  coeff_d*bi_y
                    # ....
                    det_Hess = abs(F1x*F2y-F1y*F2x)
                    # ....
                    comp_1         = ( F2y*ux - F2x*uy)/det_Hess * F2y #/sqrt(F1y**2+ F2y**2)
                    comp_2         = (-F1y*ux + F1x*uy)/det_Hess * F1y #/sqrt(F1y**2+ F2y**2)
                    # ...
                    lvalues_u2[g2]  = u * sqrt(F1y**2+ F2y**2)
                    lvalues_ux[g2] = -(comp_1 + comp_2) #* sqrt(F1y**2+ F2y**2)
                                                
           for il_2 in range(0, p2+1):
                i2 = i_span_2 - p2 + il_2

                v = 0.0
                for g2 in range(0, k2):
                    bi_0     =  basis_2[ie2, il_2, 0, g2]
                    wsurf    =  weights_2[ie2, g2]
                
                    #.. 
                    v       += bi_0 * (lvalues_ux[g2]+S_DDM * lvalues_u2[g2]) * wsurf
                    
                rhs[ne1+2*p1-1,i2+p2] += v
    else :
        
        ie1      = ne1 -1
        i_span_3  = spans_3[ie1]
        for ie2 in range(0, ne2):
           i_span_2 = spans_2[ie2]
           
           for g2 in range(0, k2):               
                    #... here we use the information from the second domain
                    lcoeffs_m1[ : , : ] = vector_m3[i_span_3 : i_span_3+p3+1, i_span_2 : i_span_2+p2+1]
                    lcoeffs_m2[ : , : ] = vector_m4[i_span_3 : i_span_3+p3+1, i_span_2 : i_span_2+p2+1]
                    #... correspond to the solution of the second domain
                    lcoeffs_di[ : , : ] = vector_np[i_span_3 : i_span_3+p3+1, i_span_2 : i_span_2+p2+1]
                    # ...
                    u    = 0.
                    ux   = 0.
                    uy   = 0.
                    F1y  = 0.
                    F2y  = 0.
                    F1x  = 0.
                    F2x  = 0.
                    for jl_2 in range(0, p4+1):
                        bi_0       = basis_4[ie2, jl_2, 0, g2]
                        bi_y       = basis_4[ie2, jl_2, 1, g2]
                        # ...
                        coeff_m1   = lcoeffs_m1[p3, jl_2]
                        coeff_m2   = lcoeffs_m2[p3, jl_2]
                        coeff_m3   = lcoeffs_m1[p3-1, jl_2]
                        coeff_m4   = lcoeffs_m2[p3-1, jl_2]

                        F1y       +=  coeff_m1 * bj_y
                        F2y       +=  coeff_m2 * bj_y
                        F1x       +=  (coeff_m1-coeff_m3) * bj_0 * p3 / knots_3[p3+1]
                        F2x       +=  (coeff_m2-coeff_m4) * bj_0 * p3 / knots_3[p3+1]

                        coeff_d   = lcoeffs_di[p3, jl_2]
                        coeff_d2  = lcoeffs_di[p3-1, jl_2]
                        # ...
                        u        +=  coeff_d*bi_0
                        ux       +=  (coeff_d-coeff_d2)*bi_0 * p3 / knots_3[p3+1]
                        uy       +=  coeff_d*bi_y
                    # ....
                    det_Hess = abs(F1x*F2y-F1y*F2x)
                    # ....
                    comp_1         = ( F2y*ux - F2x*uy)/det_Hess * F2y #/sqrt(F1y**2+ F2y**2)
                    comp_2         = -1*(-F1y*ux + F1x*uy)/det_Hess * F1y #/sqrt(F1y**2+ F2y**2)
                    # ...
                    lvalues_u2[g2] = u * sqrt(F1y**2+ F2y**2)
                    lvalues_ux[g2] = -(comp_1 + comp_2) #* sqrt(F1y**2+ F2y**2)
                                                
           for il_2 in range(0, p2+1):
                i2 = i_span_2 - p2 + il_2

                v = 0.0
                for g2 in range(0, k2):
                    bi_0     =  basis_2[ie2, il_2, 0, g2]
                    wsurf    =  weights_2[ie2, g2]
                
                    #.. 
                    v       += bi_0 * (lvalues_ux[g2]+S_DDM * lvalues_u2[g2]) * wsurf
                    
                rhs[p1,i2+p2] += v
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
