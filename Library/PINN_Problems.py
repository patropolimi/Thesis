#! /usr/bin/python3

from PINN_Wrappers import *


class Poisson_Basic(Problem_Scalar_Basic):

	""" Poisson Problem Wrapper """


	def __init__(self,ID,HL,NPL,Sigma,Domain,NResPts,NBouPts,BouLabs,SRC,Ex_Bou_D,Ex_Bou_N):
		super().__init__(ID,HL,NPL,Sigma,Domain,NResPts,NBouPts,BouLabs,SRC,Ex_Bou_D,Ex_Bou_N)
		self.Equation=self.Laplacian


	@partial(jax.jit,static_argnums=(0))
	def SingleLaplacian(self,X,W=None):

		""" Single Input Contribution/Component Network Laplacian """

		return jnp.sum(jnp.diag(self.Hessian_Network_Single(X,W)))


	@partial(jax.jit,static_argnums=(0))
	def Laplacian(self,X=None,W=None):

		""" Overall Network Laplacian Computation """

		if X is None:
			X=self.Residual_Points

		return (jax.vmap(self.SingleLaplacian,in_axes=(1,None))(X,W))[None,:]
