from simplines import compile_kernel

from simplines import SplineSpace
from simplines import TensorSpace
from simplines import StencilMatrix
from simplines import StencilVector

from simplines import build_dirichlet
#---In Poisson equation
from gallery_robin2d import assemble_vector_un_ex01
assemble_rhs         = compile_kernel(assemble_vector_un_ex01, arity=1)

import numpy                        as     np

# TODO : automate the ROBIN boundary condition
class DDM_Roben(object):
	def __init__(self, V_TOT, multipatch, domain_id, v12_mpH, S_DDM, ovlp_value):
		
        #... first computes a the mean domain
		mp1         = getGeometryMap(multipatch,0)
        xmp1, ymp1  =  mp1.RefineGeometryMap(Nelements=(V_ToT.nelements, V_ToT.nelements))
		
        self.S_DDM          = S_DDM
		self.ovlp_value     = ovlp_value
		self.sp             = V_TOT
		self.u11_mpH        = u11_mpH 
		self.u12_mpH        = u12_mpH
		self.v11_mpH        = v11_mpH 
		self.v12_mpH        = v12_mpH	
		
    def apply(self, rhs, u_next, Ndomain_id):
		
		rhsDDM   = assemble_rhs( self.V_TOT, fields = [self.u11_mpH, self.u12_mpH, self.v11_mpH, self.v12_mpH, u_next], knots = True, value = [self.domain_id, self.S_DDM, self.ovlp_value], out = rhs )
		