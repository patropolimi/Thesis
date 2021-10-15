#! /usr/bin/python3

from Basic.PINN_Wrappers import *


class Poisson_Scalar_Basic(Wrapper_Scalar_Basic):

	""" Poisson Scalar Problem Upon Basic PINN

		Input Vector -> X: Space Variable """


	def __init__(self,Architecture,Domain,Data):
		super().__init__(Architecture,Domain,Data)
		self.Equation=self.Laplacian


	@partial(jax.jit,static_argnums=(0))
	def SingleLaplacian(self,X,W):

		""" Single Input Contribution/Component Network Laplacian """

		return jnp.sum(jnp.diag(self.Hessian_Network_Single(X,W)))


	@partial(jax.jit,static_argnums=(0))
	def Laplacian(self,X,W):

		""" Overall Network Laplacian Computation """

		return (jax.vmap(self.SingleLaplacian,in_axes=(1,None))(X,W))[None,:]


class Burgers_Scalar_Basic(Wrapper_Scalar_Basic):

	""" Burgers Scalar Problem Upon Basic PINN

		Input Vector -> [T;X]
		- T: Time Variable
		- X: 1D Space Variable """


	def __init__(self,Architecture,Domain,Data):
		super().__init__(Architecture,Domain,Data)
		self.Equation=self.Burgers_Left_Hand_Side


	@partial(jax.jit,static_argnums=(0))
	def SingleLaplacian(self,X,W):

		""" Single Input Contribution/Component Network Laplacian """

		return jnp.diag(self.Hessian_Network_Single(X,W))[1]


	@partial(jax.jit,static_argnums=(0))
	def Laplacian(self,X,W):

		""" Overall Network Laplacian Computation """

		return (jax.vmap(self.SingleLaplacian,in_axes=(1,None))(X,W))[None,:]


	@partial(jax.jit,static_argnums=(0))
	def Burgers_Left_Hand_Side(self,X,W):

		""" Overall Network Burgers Equation Left Hand Side Computation """

		Evaluations=self.Network_Multiple(X,W)
		Gradients=jax.vmap(self.Gradient_Network_Single,in_axes=(1,None),out_axes=1)(X,W)
		Gradients_T=Gradients[0,:][None,:]
		Gradients_X=Gradients[1,:][None,:]
		Laplacians=self.Laplacian(X,W)
		return (Gradients_T+Evaluations*Gradients_X-self.Data['Nu']*Laplacians)


class Heat_Scalar_Basic(Wrapper_Scalar_Basic):

	""" Heat Scalar Problem Upon Basic PINN

		Input Vector -> [T;X]
		- T: Time Variable
		- X: Space Variable """


	def __init__(self,Architecture,Domain,Data):
		super().__init__(Architecture,Domain,Data)
		self.Equation=self.Heat_Left_Hand_Side


	@partial(jax.jit,static_argnums=(0))
	def SingleLaplacian(self,X,W):

		""" Single Input Contribution/Component Network Laplacian """

		return jnp.sum(jnp.diag(self.Hessian_Network_Single(X,W))[1:])


	@partial(jax.jit,static_argnums=(0))
	def Laplacian(self,X,W):

		""" Overall Network Laplacian Computation """

		return (jax.vmap(self.SingleLaplacian,in_axes=(1,None))(X,W))[None,:]


	@partial(jax.jit,static_argnums=(0))
	def Heat_Left_Hand_Side(self,X,W):

		""" Overall Network Heat Equation Left Hand Side Computation """

		Gradients=jax.vmap(self.Gradient_Network_Single,in_axes=(1,None),out_axes=1)(X,W)
		Gradients_T=Gradients[0,:][None,:]
		Laplacians=self.Laplacian(X,W)
		return (Gradients_T-self.Data['Nu']*Laplacians)


class Wave_Scalar_Basic(Wrapper_Scalar_Basic):

	""" Wave Scalar Problem Upon Basic PINN

		Input Vector -> [T;X]
		- T: Time Variable
		- X: Space Variable """


	def __init__(self,Architecture,Domain,Data):
		super().__init__(Architecture,Domain,Data)
		self.Equation=self.Wave_Left_Hand_Side


	@partial(jax.jit,static_argnums=(0))
	def Single_Gradient_TT_And_Laplacian(self,X,W):

		""" Single Network Input Contribution/Component Of:
		 	- Second Time Derivative
			- Laplacian """

		Hessian_Diagonal=jnp.diag(self.Hessian_Network_Single(X,W))
		Gradient_TT=jnp.reshape(Hessian_Diagonal[0],(1,))
		Laplacian=jnp.reshape(jnp.sum(Hessian_Diagonal[1,:]),(1,))
		return jnp.concatenate((Gradient_TT,Laplacian),axis=0)


	@partial(jax.jit,static_argnums=(0))
	def Gradient_TT_And_Laplacian(self,X,W):

		""" Overall Network Computation Of:
		 	- Second Time Derivative
			- Laplacian """

		return (jax.vmap(self.Single_Gradient_TT_And_Laplacian,in_axes=(1,None),out_axes=1)(X,W))


	@partial(jax.jit,static_argnums=(0))
	def Wave_Left_Hand_Side(self,X,W):

		""" Overall Network Wave Equation Left Hand Side Computation """

		Second_Order_Elements=Gradient_TT_And_Laplacian(X,W)
		Gradients_TT=Second_Order_Elements[0,:][None,:]
		Laplacians=Second_Order_Elements[1,:][None,:]
		return (Gradients_TT-(self.Data['C']**2)*Laplacians)


class Allen_Cahn_Scalar_Basic(Wrapper_Scalar_Basic):

	""" Allen-Cahn Scalar Problem Upon Basic PINN

		Input Vector -> [T;X]
		- T: Time Variable
		- X: Space Variable """


	def __init__(self,Architecture,Domain,Data):
		super().__init__(Architecture,Domain,Data)
		self.Equation=self.Allen_Cahn_Left_Hand_Side


	@partial(jax.jit,static_argnums=(0))
	def SingleLaplacian(self,X,W):

		""" Single Input Contribution/Component Network Laplacian """

		return jnp.sum(jnp.diag(self.Hessian_Network_Single(X,W))[1:])


	@partial(jax.jit,static_argnums=(0))
	def Laplacian(self,X,W):

		""" Overall Network Laplacian Computation """

		return (jax.vmap(self.SingleLaplacian,in_axes=(1,None))(X,W))[None,:]


	@partial(jax.jit,static_argnums=(0))
	def Allen_Cahn_Left_Hand_Side(self,X,W):

		""" Overall Network Allen-Cahn Equation Left Hand Side Computation """

		Evaluations=self.Network_Multiple(X,W)
		Gradients=jax.vmap(self.Gradient_Network_Single,in_axes=(1,None),out_axes=1)(X,W)
		Gradients_T=Gradients[0,:][None,:]
		Laplacians=self.Laplacian(X,W)
		return (Gradients_T-self.Data['Gamma1']*Laplacians-self.Data['Gamma2']*(Evaluations-Evaluations**3))


class Helmholtz_Scalar_Basic(Wrapper_Scalar_Basic):

	""" Helmoltz Scalar Problem Upon Basic PINN

		Input Vector -> X: Space Variable """


	def __init__(self,Architecture,Domain,Data):
		super().__init__(Architecture,Domain,Data)
		self.Equation=self.Helmholtz_Left_Hand_Side


	@partial(jax.jit,static_argnums=(0))
	def SingleLaplacian(self,X,W):

		""" Single Input Contribution/Component Network Laplacian """

		return jnp.sum(jnp.diag(self.Hessian_Network_Single(X,W)))


	@partial(jax.jit,static_argnums=(0))
	def Laplacian(self,X,W):

		""" Overall Network Laplacian Computation """

		return (jax.vmap(self.SingleLaplacian,in_axes=(1,None))(X,W))[None,:]


	@partial(jax.jit,static_argnums=(0))
	def Helmholtz_Left_Hand_Side(self,X,W):

		""" Overall Network Helmholtz Equation Left Hand Side Computation """

		Evaluations=self.Network_Multiple(X,W)
		Laplacians=self.Laplacian(X,W)
		return (Laplacians+(self.Data['K']**2)*Evaluations)
