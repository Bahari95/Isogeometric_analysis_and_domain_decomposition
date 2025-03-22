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
        ie2      = ne2-1
        i_span_2 = spans_2[ie2]
        for ie1 in range(0, ne1):
            i_span_1 = spans_1[ie1]

            lcoeffs_m1[ : , : ] = vector_m1[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_m2[ : , : ] = vector_m2[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g2 in range(0, k2):

                F1y = 0.0
                F2y = 0.0
                for il_1 in range(0, p1+1):

                    bj_y     = basis_1[ie1,il_1,1,g2]
                    coeff_m1 = lcoeffs_m1[il_1, p2]
                    coeff_m2 = lcoeffs_m2[il_1, p2]

                    F1y     +=  coeff_m1 * bj_y
                    F2y     +=  coeff_m2 * bj_y

                J_mat[0 ,g2] = sqrt(F1y**2 + F2y**2)
            for il_1 in range(0, p1+1):
                for jl_1 in range(0, p2+1):
                    i1 = i_span_1 - p1 + il_1
                    j1 = i_span_1 - p2 + jl_1
                    v  = 0.0
                    for g2 in range(0, k2):
                        bi_0  = basis_1[ie1, il_1, 0, g2]
                        bj_0  = basis_1[ie1, jl_1, 0, g2]
                        wvol  = weights_1[ie1, g2] * J_mat[0, g2]

                        v    += S_DDM * bi_0 * bj_0 * wvol

                    
                    matrix[p1+i1, i_span_2+p2, p1+j1-i1, p2]  += v               
                   
    if domain_nb == 1 :
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
    if domain_nb == 2:
        ie2      = 0
        i_span_2 = spans_2[ie2]
        for ie1 in range(0, ne1):
            i_span_1 = spans_1[ie1]

            lcoeffs_m1[ : , : ] = vector_m1[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_m2[ : , : ] = vector_m2[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g2 in range(0, k2):

                F1y = 0.0
                F2y = 0.0
                for il_1 in range(0, p1+1):

                    bj_y     = basis_1[ie1,il_1,1,g2]
                    coeff_m1 = lcoeffs_m1[il_1, 0]
                    coeff_m2 = lcoeffs_m2[il_1, 0]

                    F1y     +=  coeff_m1 * bj_y
                    F2y     +=  coeff_m2 * bj_y

                J_mat[0 ,g2] = sqrt(F1y**2 + F2y**2)
            for il_1 in range(0, p1+1):
                for jl_1 in range(0, p2+1):
                    i1 = i_span_1 - p1 + il_1
                    j1 = i_span_1 - p2 + jl_1
                    v  = 0.0
                    for g2 in range(0, k2):
                        bi_0  = basis_1[ie1, il_1, 0, g2]
                        bj_0  = basis_1[ie1, jl_1, 0, g2]
                        wvol  = weights_1[ie1, g2] * J_mat[0, g2]

                        v    += S_DDM * bi_0 * bj_0 * wvol

                    
                    matrix[p1+i1, p2, p1+j1-i1, p2]  += v 
        
             
    # ...

#==============================================================================Assemble rhs Poisson
##---2 : rhs
@types(	'int', 'int','int', 'int', 'int', 'int',
	 'int', 'int', 'int', 'int', 'int', 'int',
 	'int[:]',  'int[:]',  'int[:]', 'int[:]', 'int[:]', 'int[:]',
 	'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 
 	'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 
 	'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 
 	 'float[:]', 'float[:]', 'float[:]', 'float[:]', 'float[:]', 'float[:]',
 	 'double[:,:]', 'double[:,:]', 'double[:,:]',
 	 'double[:,:]', 'double[:,:]', 'double[:,:]',
 	 'double[:,:]', 'double[:,:]', 'double[:,:]',
 	   'int', 'float', 'float', 'float', 'double[:,:]')
def assemble_vector_un_ex01(ne1, ne2, ne3, ne4, ne5, ne6,
			    p1, p2, p3, p4, p5, p6,
    			    spans_1, spans_2, spans_3, spans_4, spans_5, spans_6,
    			    basis_1, basis_2, basis_3, basis_4, basis_5, basis_6,
			    weights_1, weights_2, weights_3, weights_4, weights_5, weights_6,
			    points_1, points_2, points_3, points_4, points_5, points_6,
			    knots_1, knots_2, knots_3, knots_4, knots_5, knots_6,
			    vector_m1, vector_m2, vector_d,
                            vector_m3, vector_m4, vector_np1,
                            vector_m5, vector_m6, vector_np2,
    			    domain_nb, S_DDM, ovlp_value_left, ovlp_value_right, rhs):

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
                    # .. Test 2 Quart annnulus
                    #f =  -20.0*x*y*(800*x - 400.0)*sinh(-6.25*y**2 + 400*(x - 0.5)**2)/cosh(-6.25*y**2 + 400*(x - 0.5)**2)**2 - 1562.5*y**3*(-x**2 - y**2 + 1.0)*sinh(-6.25*y**2 + 400*(x - 0.5)**2)**2/cosh(-6.25*y**2 + 400*(x - 0.5)**2)**3 
                    #f += 781.25*y**3*(-x**2 - y**2 + 1.0)/cosh(-6.25*y**2 + 400*(x - 0.5)**2) + 250.0*y**3*sinh(-6.25*y**2 + 400*(x - 0.5)**2)/cosh(-6.25*y**2 + 400*(x - 0.5)**2)**2 - 6400000.0*y*(x - 0.5)**2*(-x**2 - y**2 + 1.0)*sinh(-6.25*y**2 + 400*(x - 0.5)**2)**2/cosh(-6.25*y**2 + 400*(x - 0.5)**2)**3 
                    #f += 5.0*y*(640000*(x - 0.5)**2)*(-x**2 - y**2 + 1.0)/cosh(-6.25*y**2 + 400*(x - 0.5)**2) + 3812.5*y*(-x**2 - y**2 + 1.0)*sinh(-6.25*y**2 + 400*(x - 0.5)**2)/cosh(-6.25*y**2 + 400*(x - 0.5)**2)**2 + 40.0*y/cosh(-6.25*y**2 + 400*(x - 0.5)**2) 
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
    #span = find_span( knots, degree, xq )
   
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Assembles Neumann Condition
    if domain_nb == 0 :
      i1_ovrlp   = ne1+2*p1-1
      neum_sign1      = -1.
      neum_sign2      =  1.
      xq = ovlp_value_right
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
		  # ...
      span_3    = span
      from numpy import sign                
      for ie2 in range(0, ne2):
		         i_span_2 = spans_2[ie2]
		         
		         for g2 in range(0, k2):
		                  # ...
		                  #... here we use the information from the second domain
		                  lcoeffs_m1[ : , : ] = vector_m3[span_3 : span_3+p3+1, i_span_2 : i_span_2+p2+1]
		                  lcoeffs_m2[ : , : ] = vector_m4[span_3 : span_3+p3+1, i_span_2 : i_span_2+p2+1]
		                  #... correspond to the solution of the second domain
		                  lcoeffs_di[ : , : ] = vector_np1[span_3 : span_3+p3+1, i_span_2 : i_span_2+p2+1]
		                   # ...
		                  u    = 0.
		                  ux   = 0.
		                  uy   = 0.
		                  x    = 0.
		                  y    = 0.
		                  F1y  = 0.
		                  F2y  = 0.
		                  F1x  = 0.
		                  F2x  = 0.
		                  for il_2 in range(0, p4+1):
		                      for il_1 in range(0, p3+1):
		                          # ... there is multiplication by zero that can be optimized
		                          bi_x      = basis3[1,il_1] * basis_4[ie2, il_2, 0, g2]
		                          bi_0      = basis3[0,il_1] * basis_4[ie2, il_2, 0, g2]
		                          bi_y      = basis3[0,il_1] * basis_4[ie2, il_2, 1, g2]
		                          # ...
		                          coeff_m1   = lcoeffs_m1[il_1, il_2]
		                          coeff_m2   = lcoeffs_m2[il_1, il_2]

		                          x         +=  coeff_m1 * bi_0
		                          y         +=  coeff_m2 * bi_0
		                          F1y       +=  coeff_m1 * bi_y
		                          F2y       +=  coeff_m2 * bi_y
		                          F1x       +=  coeff_m1 * bi_x
		                          F2x       +=  coeff_m2 * bi_x

		                          coeff_d   = lcoeffs_di[il_1, il_2]
		                          # ...
		                          u        +=  coeff_d * bi_0
		                          ux       +=  coeff_d * bi_x
		                          uy       +=  coeff_d*bi_y
		                  # ....
		                  det_Hess        = abs(F1x*F2y-F1y*F2x)
		                  # ...
		                  comp_1          = neum_sign1 * ( F2y*ux - F2x*uy)/det_Hess * F2y #/sqrt(F1y**2+ F2y**2)
		                  comp_2          = neum_sign2 * (-F1y*ux + F1x*uy)/det_Hess * F1y #/sqrt(F1y**2+ F2y**2)
		                  # ...
		                  lvalues_u2[g2]  = u * sqrt(F1y**2+ F2y**2)
		                  lvalues_ux[g2]  = -(comp_1 + comp_2) #* sqrt(F1y**2+ F2y**2)
		                                              
		         for il_2 in range(0, p2+1):
		               i2 = i_span_2 - p2 + il_2

		               v = 0.0
		               for g2 in range(0, k2):
		                     bi_0     =  basis_2[ie2, il_2, 0, g2]
		                     wsurf    =  weights_2[ie2, g2]
		                     #.. 
		                     v       += bi_0 * (lvalues_ux[g2]+S_DDM * lvalues_u2[g2]) * wsurf
		                    
		               rhs[i1_ovrlp,i2+p2] += v
				    
      xq = ovlp_value_left
		  #~~~~~~~~~~~~~~~
		  #span = find_span( knots, degree, xq )
		  #~~~~~~~~~~~~~~~
		  # Knot index at left/right boundary
      low  = degree
      high = len(knots_6)-1-degree
		  # Check if point is exactly on left/right boundary, or outside domain
      if xq <= knots_6[low ]: 
		          span = low
      if xq >= knots_6[high]: 
		          span = high-1
      else :
		      # Perform binary search
		      span = (low+high)//2
		      while xq < knots_6[span] or xq >= knots_6[span+1]:
		          if xq < knots_6[span]:
		              high = span
		          else:
		              low  = span
		          span = (low+high)//2
      ndu[0,0] = 1.0
      for j in range(0,degree):
		      left [j] = xq - knots_6[span-j]
		      right[j] = knots_6[span+1+j] - xq
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
		  # ...
      span_3    = span
		  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		  # Assembles Neumann Condition
      i2_ovrlp   = ne2+2*p2-1
		    
      neum_sign1      =  -1.
      neum_sign2      =  1.
		 

      for ie1 in range(0, ne1):
		         i_span_1 = spans_1[ie1]
		         for g2 in range(0, k2):
		                  # ...
		                  #... here we use the information from the second domain
		                  lcoeffs_m1[ : , : ] = vector_m5[i_span_1 : i_span_1+p1+1, span_3 : span_3+p3+1]
		                  lcoeffs_m2[ : , : ] = vector_m6[i_span_1 : i_span_1+p1+1, span_3 : span_3+p3+1]
		                  #... correspond to the solution of the second domain
		                  #... correspond to the solution of the second domain
		                  lcoeffs_di[ : , : ] =vector_np2[i_span_1 : i_span_1+p1+1, span_3 : span_3+p3+1]
		                   # ...
		                  u    = 0.
		                  ux   = 0.
		                  uy   = 0.
		                  x    = 0.
		                  y    = 0.
		                  F1y  = 0.
		                  F2y  = 0.
		                  F1x  = 0.
		                  F2x  = 0.
		                  for il_1 in range(0, p4+1):
		                      for il_2 in range(0, p3+1):
		                          # ... there is multiplication by zero that can be optimized
		                          bi_x      = basis3[1,il_1] * basis_5[ie1, il_2, 0, g2]
		                          bi_0      = basis3[0,il_1] * basis_5[ie1, il_2, 0, g2]
		                          bi_y      = basis3[0,il_1] * basis_5[ie1, il_2, 1, g2]
		                          # ...
		                          coeff_m1   = lcoeffs_m1[il_2, il_1]
		                          coeff_m2   = lcoeffs_m2[il_2, il_1]

		                          x         +=  coeff_m1 * bi_0
		                          y         +=  coeff_m2 * bi_0
		                          F1y       +=  coeff_m1 * bi_y
		                          F2y       +=  coeff_m2 * bi_y
		                          F1x       +=  coeff_m1 * bi_x
		                          F2x       +=  coeff_m2 * bi_x

		                          coeff_d   = lcoeffs_di[il_2, il_1]
		                          # ...
		                          u        +=  coeff_d * bi_0
		                          ux       +=  coeff_d * bi_x
		                          uy       +=  coeff_d*bi_y
		         
		        
		                  # ....
		                  det_Hess        = abs(F1x*F2y-F1y*F2x)
		                  # ...
		                  comp_1          = neum_sign1 * ( F2y*ux - F2x*uy)/det_Hess * F2y #/sqrt(F1y**2+ F2y**2)
		                  comp_2          = neum_sign2 * (-F1y*ux + F1x*uy)/det_Hess * F1y #/sqrt(F1y**2+ F2y**2)
		                  # ...
		                  lvalues_u2[g2]  = u * sqrt(F1y**2+ F2y**2)
		                  lvalues_ux[g2]  = -(comp_1 + comp_2) #* sqrt(F1y**2+ F2y**2)
		                                              
		         for il_1 in range(0, p1+1):
		               i1 = i_span_1 - p1 + il_1

		               v = 0.0
		               for g2 in range(0, k2):
		                     bi_0     =  basis_1[ie1, il_1, 0, g2]
		                     wsurf    =  weights_1[ie1, g2]
		                     #.. 
		                     v       += bi_0 * (lvalues_ux[g2]+S_DDM * lvalues_u2[g2]) * wsurf
		               rhs[i1+p2, i2_ovrlp] += v
    if domain_nb == 1 :
      xq = ovlp_value_left
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
		  # ...
      span_3    = span
	 
      i1_ovrlp   = p1
      neum_sign1      =  1.
      neum_sign2      = -1.
		                
      for ie2 in range(0, ne2):
		         i_span_2 = spans_2[ie2]
		         
		         for g2 in range(0, k2):
		                  # ...
		                  #... here we use the information from the second domain
		                  lcoeffs_m1[ : , : ] = vector_m3[span_3 : span_3+p3+1, i_span_2 : i_span_2+p2+1]
		                  lcoeffs_m2[ : , : ] = vector_m4[span_3 : span_3+p3+1, i_span_2 : i_span_2+p2+1]
		                  #... correspond to the solution of the second domain
		                  lcoeffs_di[ : , : ] = vector_np1[span_3 : span_3+p3+1, i_span_2 : i_span_2+p2+1]
		                   # ...
		                  u    = 0.
		                  ux   = 0.
		                  uy   = 0.
		                  x    = 0.
		                  y    = 0.
		                  F1y  = 0.
		                  F2y  = 0.
		                  F1x  = 0.
		                  F2x  = 0.
		                  for il_2 in range(0, p4+1):
		                      for il_1 in range(0, p3+1):
		                          # ... there is multiplication by zero that can be optimized
		                          bi_x      = basis3[1,il_1] * basis_4[ie2, il_2, 0, g2]
		                          bi_0      = basis3[0,il_1] * basis_4[ie2, il_2, 0, g2]
		                          bi_y      = basis3[0,il_1] * basis_4[ie2, il_2, 1, g2]
		                          # ...
		                          coeff_m1   = lcoeffs_m1[il_1, il_2]
		                          coeff_m2   = lcoeffs_m2[il_1, il_2]

		                          x         +=  coeff_m1 * bi_0
		                          y         +=  coeff_m2 * bi_0
		                          F1y       +=  coeff_m1 * bi_y
		                          F2y       +=  coeff_m2 * bi_y
		                          F1x       +=  coeff_m1 * bi_x
		                          F2x       +=  coeff_m2 * bi_x

		                          coeff_d   = lcoeffs_di[il_1, il_2]
		                          # ...
		                          u        +=  coeff_d * bi_0
		                          ux       +=  coeff_d * bi_x
		                          uy       +=  coeff_d*bi_y
		                  # ....
		                  det_Hess        = abs(F1x*F2y-F1y*F2x)
		                  # ...
		                  comp_1          = neum_sign1 * ( F2y*ux - F2x*uy)/det_Hess * F2y #/sqrt(F1y**2+ F2y**2)
		                  comp_2          = neum_sign2 * (-F1y*ux + F1x*uy)/det_Hess * F1y #/sqrt(F1y**2+ F2y**2)
		                  # ...
		                  lvalues_u2[g2]  = u * sqrt(F1y**2+ F2y**2)
		                  lvalues_ux[g2]  = -(comp_1 + comp_2) #* sqrt(F1y**2+ F2y**2)
		                                              
		         for il_2 in range(0, p2+1):
		               i2 = i_span_2 - p2 + il_2

		               v = 0.0
		               for g2 in range(0, k2):
		                     bi_0     =  basis_2[ie2, il_2, 0, g2]
		                     wsurf    =  weights_2[ie2, g2]
		                     #.. 
		                     v       += bi_0 * (lvalues_ux[g2]+S_DDM * lvalues_u2[g2]) * wsurf
		                    
		               rhs[i1_ovrlp,i2+p2] += v
    if domain_nb == 2 :              
      xq = ovlp_value_left
		  #~~~~~~~~~~~~~~~
		  #span = find_span( knots, degree, xq )
		  #~~~~~~~~~~~~~~~
		  # Knot index at left/right boundary
      low  = degree
      high = len(knots_4)-1-degree
		  # Check if point is exactly on left/right boundary, or outside domain
      if xq <= knots_4[low ]: 
		          span = low
      if xq >= knots_4[high]: 
		          span = high-1
      else :
		      # Perform binary search
		      span = (low+high)//2
		      while xq < knots_4[span] or xq >= knots_4[span+1]:
		          if xq < knots_4[span]:
		              high = span
		          else:
		              low  = span
		          span = (low+high)//2
      ndu[0,0] = 1.0
      for j in range(0,degree):
		      left [j] = xq - knots_4[span-j]
		      right[j] = knots_4[span+1+j] - xq
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
		  # ...
      span_3    = span
		  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	  
      i2_ovrlp   = p2
      neum_sign1      =  1.
      neum_sign2      = -1.
		             
      for ie1 in range(0, ne1):
		         i_span_1 = spans_1[ie1]
		         for g2 in range(0, k2):
		                  # ...
		                  #... here we use the information from the second domain
		                  lcoeffs_m1[ : , : ] = vector_m3[i_span_1 : i_span_1+p1+1, span_3 : span_3+p3+1]
		                  lcoeffs_m2[ : , : ] = vector_m4[i_span_1 : i_span_1+p1+1, span_3 : span_3+p3+1]
		                  #... correspond to the solution of the second domain
		                  #... correspond to the solution of the second domain
		                  lcoeffs_di[ : , : ] =vector_np1[i_span_1 : i_span_1+p1+1, span_3 : span_3+p3+1]
		                   # ...
		                  u    = 0.
		                  ux   = 0.
		                  uy   = 0.
		                  x    = 0.
		                  y    = 0.
		                  F1y  = 0.
		                  F2y  = 0.
		                  F1x  = 0.
		                  F2x  = 0.
		                  for il_1 in range(0, p4+1):
		                      for il_2 in range(0, p3+1):
		                          # ... there is multiplication by zero that can be optimized
		                          bi_x      = basis3[1,il_1] * basis_3[ie1, il_2, 0, g2]
		                          bi_0      = basis3[0,il_1] * basis_3[ie1, il_2, 0, g2]
		                          bi_y      = basis3[0,il_1] * basis_3[ie1, il_2, 1, g2]
		                          # ...
		                          coeff_m1   = lcoeffs_m1[il_2, il_1]
		                          coeff_m2   = lcoeffs_m2[il_2, il_1]

		                          x         +=  coeff_m1 * bi_0
		                          y         +=  coeff_m2 * bi_0
		                          F1y       +=  coeff_m1 * bi_y
		                          F2y       +=  coeff_m2 * bi_y
		                          F1x       +=  coeff_m1 * bi_x
		                          F2x       +=  coeff_m2 * bi_x

		                          coeff_d   = lcoeffs_di[il_2, il_1]
		                          # ...
		                          u        +=  coeff_d * bi_0
		                          ux       +=  coeff_d * bi_x
		                          uy       +=  coeff_d*bi_y
		         
		        
		                  # ....
		                  det_Hess        = abs(F1x*F2y-F1y*F2x)
		                  # ...
		                  comp_1          = neum_sign1 * ( F2y*ux - F2x*uy)/det_Hess * F2y #/sqrt(F1y**2+ F2y**2)
		                  comp_2          = neum_sign2 * (-F1y*ux + F1x*uy)/det_Hess * F1y #/sqrt(F1y**2+ F2y**2)
		                  # ...
		                  lvalues_u2[g2]  = u * sqrt(F1y**2+ F2y**2)
		                  lvalues_ux[g2]  = -(comp_1 + comp_2) #* sqrt(F1y**2+ F2y**2)
		                                              
		         for il_1 in range(0, p1+1):
		               i1 = i_span_1 - p1 + il_1

		               v = 0.0
		               for g2 in range(0, k2):
		                     bi_0     =  basis_1[ie1, il_1, 0, g2]
		                     wsurf    =  weights_1[ie1, g2]
		                     #.. 
		                     v       += bi_0 * (lvalues_ux[g2]+S_DDM * lvalues_u2[g2]) * wsurf
		                    
		               rhs[i1+p2, i2_ovrlp] += v                    
#=================================================================================
# norm in uniform mesh norm
@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]',  'double[:,:]',  'double[:,:]')
def assemble_norm_ex01(ne1, ne2, p1, p2, spans_1, spans_2,  basis_1, basis_2,  weights_1, weights_2, points_1, points_2, vector_m1, vector_m2, vector_u, rhs):

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
    lcoeffs_m1 = zeros((p1+1,p2+1))
    lcoeffs_m2 = zeros((p1+1,p2+1))
    lcoeffs_u  = zeros((p1+1,p2+1))
    # ...
    lvalues_ux = zeros((k1, k2))
    lvalues_uy = zeros((k1, k2))
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
            lcoeffs_m1[ : , : ] = vector_m1[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_m2[ : , : ] = vector_m2[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    x   = 0.0
                    y   = 0.0
                    F1x = 0.0
                    F1y = 0.0
                    F2x = 0.0
                    F2y = 0.0
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

                    det_J = abs(F1x*F2y-F1y*F2x)
                    
                    # ...                              
                    wvol  = weights_1[ie1, g1] * weights_2[ie2, g2]
                    x1    =  points_1[ie1, g1]
                    x2    =  points_2[ie2, g2]

                    uh    = lvalues_u[g1,g2]
                    sx    = lvalues_ux[g1,g2]
                    sy    = lvalues_uy[g1,g2]
                    #... TEST 2 Quart annulus
                    #f    = 5.0/cosh(50 * ((8*(x-0.5)**2) -y**2* 0.125))*(1.-x**2-y**2)*y
                    #fx   = -10.0*x*y/cosh(-6.25*y**2 + 400*(x - 0.5)**2) - 5.0*y*(800*x - 400.0)*(-x**2 - y**2 + 1.0)*sinh(-6.25*y**2 + 400*(x - 0.5)**2)/cosh(-6.25*y**2 + 400*(x - 0.5)**2)**2 
                    #fy   =  62.5*y**2*(-x**2 - y**2 + 1.0)*sinh(-6.25*y**2 + 400*(x - 0.5)**2)/cosh(-6.25*y**2 + 400*(x - 0.5)**2)**2 - 10.0*y**2/cosh(-6.25*y**2 + 400*(x - 0.5)**2) + 5.0*(-x**2 - y**2 + 1.0)/cosh(-6.25*y**2 + 400*(x - 0.5)**2)  
                    #... 
                    f    = sin(2.*pi*x)*sin(2.*pi*y)
                    fx   = 2.*pi*cos(2.*pi*x)*sin(2.*pi*y)
                    fy   = 2.*pi*sin(2.*pi*x)*cos(2.*pi*y)
                    # ...
                    uhx   = (F2y*sx-F2x*sy)/det_J
                    uhy   = (F1x*sy-F1y*sx)/det_J

                    w    += ((uhx-fx)**2 +(uhy-fy)**2)* wvol * det_J
                    v    += (uh-f)**2 * wvol * det_J

            error_H1      += w
            error_l2      += v
    rhs[p1,p2]   = sqrt(error_l2)
    rhs[p1,p2+1] = sqrt(error_H1)
    #...
