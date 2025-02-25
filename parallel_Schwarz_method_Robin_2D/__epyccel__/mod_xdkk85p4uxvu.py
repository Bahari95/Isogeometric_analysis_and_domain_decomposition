@types(	'int', 'int','int', 'int', 'int', 'int', 'int', 'int', 'int[:]',  'int[:]',  'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'int', 'real', 'real','real[:]', 'real[:]', 'real[:]', 'real[:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]')
def assemble_vector_un_ex01(ne1, ne2, ne3, ne4,
			    p1, p2, p3, p4, 
    			    spans_1, spans_2, spans_3, spans_4,
    			    basis_1, basis_2, basis_3, basis_4,
    			    weights_1, weights_2, weights_3, weights_4,
    			    points_1, points_2, points_3, points_4,
    			    domain_nb, S_DDM, ovlp_value,
    			    knots_1, knots_2, knots_3, knots_4,
    			    vector_m1, vector_m2, vector_d, rhs):

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
    arr_J_mat0  = zeros((k1,k2))
    arr_J_mat1  = zeros((k1,k2))
    arr_J_mat2  = zeros((k1,k2))
    arr_J_mat3  = zeros((k1,k2))
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
    lvalues_u2      = zeros(k2)
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
      neum_sign = 1.
    else :
      i1_ovrlp  = p1
      neum_sign = -1.
    span_3 = span
                
    for ie2 in range(0, ne2):
           i_span_2 = spans_2[ie2]
           
           for g2 in range(0, k2):               
                    
                    lcoeffs_m1[ : , : ] = vector_m1[span_3 : span_3+p3+1, i_span_2 : i_span_2+p2+1]
                    lcoeffs_m2[ : , : ] = vector_m2[span_3 : span_3+p3+1, i_span_2 : i_span_2+p2+1]
                    lcoeffs_di[ : , : ] = vector_d[ span_3 : span_3+p3+1, i_span_2 : i_span_2+p2+1]
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
                             coeff_d   = lcoeffs_di[il_1, il_2]
                             # ...
                             u        +=  coeff_d*bi_0
                             ux       +=  coeff_d*bi_x
                             uy       +=  coeff_d*bi_y
                    # ....
                    x1             = ovlp_value
                    x2             = points_2[ie2,g2]

                    F1             = ( r_min+delta_r*x1)*cos(theta*x2)
                    F1y            = -theta* ( r_min+delta_r*x1)*sin(theta*x2)
                    F1x            = delta_r*cos(theta*x2)
                    F1xx           = 0.0
                    F1yy           = -(theta)**2 * ( r_min+delta_r*x1)*cos(theta*x2)
                    F1xy           = -theta*delta_r*sin(theta*x2)

                    F2             = ( r_min+delta_r*x1)*sin(theta*x2)
                    F2x            = delta_r*sin(theta*x2)
                    F2y            = theta*( r_min+delta_r*x1)*cos(theta*x2)
                    F2xx           = 0.0
                    F2yy           = -(theta)**2*( r_min+delta_r*x1)*sin(theta*x2)
                    F2xy           = theta*delta_r*cos(theta*x2)

                    det_Hess       = abs(F1x*F2y-F1y*F2x)
                    
                    tang_1         =  (F1yy*(F1y**2+ F2y**2) - F1y*(F1yy*F1y + F2yy*F2y) )/(sqrt(F1y**2+ F2y**2)**3)
                    tang_2         =  (F2yy*(F1y**2+ F2y**2) - F2y*(F1yy*F1y + F2yy*F2y) )/(sqrt(F1y**2+ F2y**2)**3)
                    # ...
                    comp_1         = ( F2y*ux - F2x*uy)/det_Hess * tang_1/sqrt(tang_1**2+tang_2**2)
                    comp_2         = (-F1y*ux + F1x*uy)/det_Hess * tang_2/sqrt(tang_1**2+tang_2**2)
                    # ...
                    lvalues_u2[g2]  = u * det_Hess
                    lvalues_ux[g2] = -(comp_1 + comp_2)* sqrt(F1y**2+ F2y**2)
                                                
           for il_2 in range(0, p2+1):
                 i2 = i_span_2 - p2 + il_2

                 v = 0.0
                 for g2 in range(0, k2):
                       bi_0     =  basis_2[ie2, il_2, 0, g2]
                       wsurf    =  weights_2[ie2, g2]
                  
                       #.. 
                       v       += bi_0 * (lvalues_ux[g2]+S_DDM * lvalues_u2[g2]) * wsurf
                      
                 rhs[i1_ovrlp,i2+p2] += v
    # ...
