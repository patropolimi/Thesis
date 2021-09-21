#! /usr/bin/python3

from PINN_Wrappers import *


class Poisson_Scalar_Basic(Problem_Scalar_Basic):

	""" Poisson Scalar Problem Wrapper """


	def __init__(self,ID,HL,NPL,Sigma,Domain,NResPts,NBouPts,BouLabs,SRC,Ex_Bou_D,Ex_Bou_N):
		super().__init__(ID,HL,NPL,Sigma,Domain,NResPts,NBouPts,BouLabs,SRC,Ex_Bou_D,Ex_Bou_N)
		self.Equation=self.Laplacian


	@partial(jax.jit,static_argnums=(0))
	def SingleLaplacian(self,X,W):

		""" Single Input Contribution/Component Network Laplacian """

		return jnp.sum(jnp.diag(self.Hessian_Network_Single(X,W)))


	@partial(jax.jit,static_argnums=(0))
	def Laplacian(self,X,W):

		""" Overall Network Laplacian Computation """

		return (jax.vmap(self.SingleLaplacian,in_axes=(1,None))(X,W))[None,:]


class Burgers_Scalar_Basic(Problem_Scalar_Basic):

	""" Burgers Scalar Problem Wrapper

		Input Vector -> [T;X]
		- T: Time Variable
		- X: 1D Space Variable """


	def __init__(self,HL,NPL,Sigma,Domain,NResPts,NBouPts,BouLabs,SRC,Ex_Bou_D,Ex_Bou_N,Nu):
		super().__init__(2,HL,NPL,Sigma,Domain,NResPts,NBouPts,BouLabs,SRC,Ex_Bou_D,Ex_Bou_N)
		self.Nu=Nu
		self.Equation=self.Burgers_Left_Hand_Side


	@partial(jax.jit,static_argnums=(0))
	def SingleGradient_XX(self,X,W):

		""" Single Input Contribution/Component Network Laplacian """

		return jnp.diag(self.Hessian_Network_Single(X,W))[1]


	@partial(jax.jit,static_argnums=(0))
	def Gradient_XX(self,X,W):

		""" Overall Network Laplacian Computation """

		return (jax.vmap(self.SingleGradient_XX,in_axes=(1,None))(X,W))[None,:]


	@partial(jax.jit,static_argnums=(0))
	def Burgers_Left_Hand_Side(self,X,W):

		""" Overall Network Burgers Equation Left Hand Side Computation """

		Evaluations=self.Network_Multiple(X,W)
		Gradients=jax.vmap(self.Gradient_Network_Single,in_axes=(1,None),out_axes=1)(X,W)
		Gradients_T=Gradients[0,:][None,:]
		Gradients_X=Gradients[1,:][None,:]
		Gradients_XX=self.Gradient_XX(X,W)
		return (Gradients_T+Evaluations*Gradients_X-self.Nu*Gradients_XX)
