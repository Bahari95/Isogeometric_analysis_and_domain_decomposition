__all__ = ['assemble_matrix_ex01',
           'assemble_vector_ex01',
           'assemble_norm_ex01',
           'assemble_matrix_ex02',
           'assemble_vector_ex02',
           'assemble_norm_ex02'
]

from pyccel.decorators import types

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

#==============================================================================Assemble rhs Poisson
#---1 : In uniform mesh
@types('int', 'int', 'int', 'int','int', 'int', 'int[:]', 'int[:]', 'int[:]', 'double[:,:,:,:]',  'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]','double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'real[:]', 'real[:]', 'real[:]', 'double[:]', 'double[:]', 'real', 'real', 'real', 'int', 'double[:]')
def assemble_vector_ex01(ne1, ne2, ne3, p1, p2, p3, spans_1, spans_2, spans_3,
		         basis_1, basis_2, basis_3, weights_1, weights_2, weights_3, 
		         points_1, points_2, points_3, knots_1, knots_2, knots_3,
		         vector_d1, vector_d2, ovlp_value_left, ovlp_value_right, S_DDM, domain_nb, rhs):

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

    # ... build rhs
    for ie1 in range(0, ne1):
            i_span_1 = spans_1[ie1]
            for il_1 in range(0, p1+1):
                    i1 = i_span_1 - p1 + il_1

                    v = 0.0
                    for g1 in range(0, k1):
                            
                            bi_0 = basis_1[ie1, il_1, 0, g1]

                            wvol  = weights_1[ie1, g1]
                            x  = points_1[ie1, g1]
                            # .. Test 0
                            u    = pi**2*sin(pi*x)
                            # ... test 1                                 
                            #u  = -(15625.0 - 31250.0*x)*(2*x - 1.0)*exp(-7812.5*((x - 0.5)**2 + (y - 0.5)**2 - 0.096)**2) 
                            #u += 7812.5*(15625.0 - 31250.0*x)*(4*x - 2.0)*((x - 0.5)**2 + (y - 0.5)**2 - 0.096)**2*exp(-7812.5*((x - 0.5)**2 + (y - 0.5)**2 - 0.096)**2) 
                            #u += -(15625.0 - 31250.0*y)*(2*y - 1.0)*exp(-7812.5*((x - 0.5)**2 + (y - 0.5)**2 - 0.096)**2) 
                            #u += 7812.5*(15625.0 - 31250.0*y)*(4*y - 2.0)*((x - 0.5)**2 + (y - 0.5)**2 - 0.096)**2*exp(-7812.5*((x - 0.5)**2 + (y - 0.5)**2 - 0.096)**2) 
                            #u += -2*(-31250.0*(x - 0.5)**2 - 31250.0*(y - 0.5)**2 + 3000.0)*exp(-7812.5*((x - 0.5)**2 + (y - 0.5)**2 - 0.096)**2)
                            # ..
                            v += bi_0 * u * wvol

                    rhs[i1+p1] += v  

    # ...
    lcoeffs_d1   = zeros(p2+1)
    lcoeffs_d2   = zeros(p3+1)
    #---Computes All basis in at the interface of the overllape points
    nders          = 1
    degree         = p2
    #..
    left           = empty( degree )
    right          = empty( degree )
    a              = empty( (       2, degree+1) )
    ndu            = empty( (degree+1, degree+1) )
    ders           = zeros( (     nders+1, degree+1) ) # output array
    #...
    basis2         = zeros((nders+1, degree+1))
    xq = ovlp_value_left
    #~~~~~~~~~~~~~~~
    low  = degree
    high = len(knots_2)-1-degree
    # Check if point is exactly on left/right boundary, or outside domain
    if xq <= knots_2[low ]: 
         span = low
    if xq >= knots_2[high]: 
         span = high-1
    else :
      # Perform binary search
      span = (low+high)//2
      while xq < knots_2[span] or xq >= knots_2[span+1]:
         if xq < knots_2[span]:
             high = span
         else:
             low  = span
         span = (low+high)//2
    ndu[0,0] = 1.0
    for j in range(0,degree):
        left [j] = xq - knots_2[span-j]
        right[j] = knots_2[span+1+j] - xq
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
    basis2[0,:] = ders[0,:]
    basis2[1,:] = ders[1,:]
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Assembles Neumann Condition
   
    if domain_nb == 0 :
      i1_ovrlp  = ne1+2*p1-1
      neum_sign = 1.

      # ...
      lcoeffs_d1[:] = vector_d1[span : span+p2+1]
      # ...
      u   = 0.
      ux  = 0.
      for il_1 in range(0, p2+1):
      	 bi_0      = basis2[0,il_1]
      	 bi_x      = basis2[1,il_1]
	     # ...
      	 coeff_d   = lcoeffs_d1[il_1]
	     # ...
      	 u        +=  coeff_d*bi_0
      	 ux       +=  coeff_d*bi_x
             	# ...
      rhs[i1_ovrlp] += neum_sign*ux+S_DDM*u
    # ...
    elif domain_nb == 1  :
      i1_ovrlp  = p1
      neum_sign = -1.
      # ...
      lcoeffs_d1[:] = vector_d1[span : span+p2+1]
      # ...
      u   = 0.
      ux  = 0.
      for il_1 in range(0, p2+1):
      	 bi_0      = basis2[0,il_1]
      	 bi_x      = basis2[1,il_1]
	     # ...
      	 coeff_d   = lcoeffs_d1[il_1]
	     # ...
      	 u        +=  coeff_d*bi_0
      	 ux       +=  coeff_d*bi_x
             	# ...
      rhs[i1_ovrlp] += neum_sign*ux+S_DDM*u
    else:   
	    i1_ovrlp  = p1

	    neum_sign = -1.
	    # ...
	    lcoeffs_d1[:] = vector_d1[span : span+p2+1]
	    # ...
	    u   = 0.
	    ux  = 0.
	    for il_1 in range(0, p2+1):
		     bi_0      = basis2[0,il_1]
		     bi_x      = basis2[1,il_1]
		     # ...
		     coeff_d   = lcoeffs_d1[il_1]
		     # ...
		     u        +=  coeff_d*bi_0
		     ux       +=  coeff_d*bi_x
		     	# ...
	    rhs[i1_ovrlp] += neum_sign*ux+S_DDM*u  
	    degree         = p3
	    #..
	    '''
	    left           = empty( degree )
	    right          = empty( degree )
	    a              = empty( (       2, degree+1) )
	    ndu            = empty( (degree+1, degree+1) )
	    ders           = zeros( (     nders+1, degree+1) ) # output array
	    '''
	    #...
	    basis3         = zeros((nders+1, degree+1))
	    xq = ovlp_value_right
	    #~~~~~~~~~~~~~~~
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
	    i1_ovrlp  = ne1+2*p1-1
	    neum_sign = 1.
	    # ...
	    lcoeffs_d2[:] = vector_d2[span : span+p3+1]
	    # ...
	    u   = 0.
	    ux  = 0.
	    for il_1 in range(0, p3+1):
		     bi_0      = basis3[0,il_1]
		     bi_x      = basis3[1,il_1]
		     # ...
		     coeff_d   = lcoeffs_d2[il_1]
		     # ...
		     u        +=  coeff_d*bi_0
		     ux       +=  coeff_d*bi_x
		     	# ..
	    rhs[i1_ovrlp] += neum_sign*ux+S_DDM*u  

#==============================================================================Assemble l2 and H1 error norm
#---1 : In uniform mesh
@types('int', 'int', 'int[:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:]', 'double[:]')
def assemble_norm_ex01(ne1, p1, spans_1, basis_1, weights_1, points_1, vector_u, rhs):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import sqrt
    from numpy import zeros

    # ... sizes
    k1 = weights_1.shape[1]
    # ...

    lcoeffs_u  = zeros(p1+1)
    lvalues_u  = zeros(k1)
    lvalues_ux = zeros(k1)

    norm_l2 = 0.
    norm_H1 = 0.
    # ...
    for ie1 in range(0, ne1):
            i_span_1 = spans_1[ie1]

            lvalues_u[ : ]  = 0.0
            lvalues_ux[ : ] = 0.0
            lcoeffs_u[ : ]  = vector_u[i_span_1 : i_span_1+p1+1]
            for il_1 in range(0, p1+1):
                    coeff_u = lcoeffs_u[il_1]

                    for g1 in range(0, k1):
                            b1  = basis_1[ie1,il_1,0,g1]
                            db1 = basis_1[ie1,il_1,1,g1]

                            lvalues_u[g1]  += coeff_u * b1
                            lvalues_ux[g1] += coeff_u * db1

            v = 0.0
            w = 0.0
            for g1 in range(0, k1):
                    wvol = weights_1[ie1, g1]

                    x    = points_1[ie1, g1]

                    # ... test 0
                    u   = sin(pi*x)
                    ux  = pi*cos(pi*x)
                    # ... test 1
                    
                    uh  = lvalues_u[g1]
                    uhx = lvalues_ux[g1]

                    v  += (u-uh)**2 * wvol
                    w  += ((ux-uhx)**2) * wvol
            norm_l2 += v
            norm_H1 += w

    norm_l2 = sqrt(norm_l2)
    norm_H1 = sqrt(norm_H1)

    rhs[p1]   = norm_l2
    rhs[p1+1] = norm_H1
    # ...
