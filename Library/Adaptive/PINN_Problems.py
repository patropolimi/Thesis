#! /usr/bin/python3

from Adaptive.PINN_Wrappers import *


class Identity_Scalar_Adaptive(Wrapper_Scalar_Adaptive):

	""" Identity Scalar Problem Upon Adaptive PINN

		Input Vector -> X: Space-Time Variable """


	def __init__(self,Architecture,Domain,Data,Adaptivity):
		super().__init__(Architecture,Domain,Data,Adaptivity)
		self.Equation=self.Network_Multiple


class ODE_Scalar_Adaptive(Wrapper_Scalar_Adaptive):

	""" ODE Scalar Problem Upon Adaptive PINN

		Input Vector -> X: 1D Space Variable """


	def __init__(self,Architecture,Domain,Data,Adaptivity):
		super().__init__(Architecture,Domain,Data,Adaptivity)
		self.Equation=self.Gradient


	@partial(jax.jit,static_argnums=(0))
	def Gradient(self,X,W):

		""" Overall Network Gradient Computation """

		return jax.vmap(self.Gradient_Network_Single,in_axes=(1,None),out_axes=1)(X,W)


class Poisson_Scalar_Adaptive(Wrapper_Scalar_Adaptive):

	""" Poisson Scalar Problem Upon Adaptive PINN

		Input Vector -> X: Space Variable """


	def __init__(self,Architecture,Domain,Data,Adaptivity):
		super().__init__(Architecture,Domain,Data,Adaptivity)
		self.Equation=self.Laplacian


	@partial(jax.jit,static_argnums=(0))
	def SingleLaplacian(self,X,W):

		""" Single Input Contribution/Component Network Laplacian """

		return jnp.sum(jnp.diag(self.Hessian_Network_Single(X,W)))


	@partial(jax.jit,static_argnums=(0))
	def Laplacian(self,X,W):

		""" Overall Network Laplacian Computation """

		return (jax.vmap(self.SingleLaplacian,in_axes=(1,None))(X,W))[None,:]


class Advection_Diffusion_Reaction_Scalar_Adaptive(Wrapper_Scalar_Adaptive):

	""" Advection-Diffusion-Reaction Scalar Problem Upon Adaptive PINN

		Input Vector -> [T;X]

		- T: Time Variable
		- X: Space Variable """


	def __init__(self,Architecture,Domain,Data,Adaptivity):
		super().__init__(Architecture,Domain,Data,Adaptivity)
		self.Equation=self.Advection_Diffusion_Reaction_Left_Hand_Side


	@partial(jax.jit,static_argnums=(0))
	def SingleLaplacian(self,X,W):

		""" Single Input Contribution/Component Network Laplacian """

		return jnp.sum(jnp.diag(self.Hessian_Network_Single(X,W)))


	@partial(jax.jit,static_argnums=(0))
	def Laplacian(self,X,W):

		""" Overall Network Laplacian Computation """

		return (jax.vmap(self.SingleLaplacian,in_axes=(1,None))(X,W))[None,:]


	@partial(jax.jit,static_argnums=(0))
	def Advection_Diffusion_Reaction_Left_Hand_Side(self,X,W):

		""" Overall Network Advection-Diffusion-Reaction Equation Left Hand Side Computation """

		Gradients=jax.vmap(self.Gradient_Network_Single,in_axes=(1,None),out_axes=1)(X,W)
		Gradients_T=Gradients[:1,:]
		Gradients_X=Gradients[1:,:]
		Laplacians=self.Laplacian(X,W)
		Advections=jax.vmap(jnp.inner,in_axes=(None,1))(self.Data['U'],Gradients_X)[None,:]
		return (Gradients_T+Advections+self.Data['G']*self.Network_Multiple(X,W)-self.Data['D']*Laplacians)


class Burgers_Scalar_Adaptive(Wrapper_Scalar_Adaptive):

	""" Burgers Scalar Problem Upon Adaptive PINN

		Input Vector -> [T;X]
		- T: Time Variable
		- X: 1D Space Variable """


	def __init__(self,Architecture,Domain,Data,Adaptivity):
		super().__init__(Architecture,Domain,Data,Adaptivity)
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


class Heat_Scalar_Adaptive(Wrapper_Scalar_Adaptive):

	""" Heat Scalar Problem Upon Adaptive PINN

		Input Vector -> [T;X]
		- T: Time Variable
		- X: Space Variable """


	def __init__(self,Architecture,Domain,Data,Adaptivity):
		super().__init__(Architecture,Domain,Data,Adaptivity)
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


class Wave_Scalar_Adaptive(Wrapper_Scalar_Adaptive):

	""" Wave Scalar Problem Upon Adaptive PINN

		Input Vector -> [T;X]
		- T: Time Variable
		- X: Space Variable """


	def __init__(self,Architecture,Domain,Data,Adaptivity):
		super().__init__(Architecture,Domain,Data,Adaptivity)
		Handler_Accelleration(self)
		self.Equation=self.Wave_Left_Hand_Side


	@partial(jax.jit,static_argnums=(0))
	def Single_Gradient_TT_And_Laplacian(self,X,W):

		""" Single Network Input Contribution/Component Of:
		 	- Second Time Derivative
			- Laplacian """

		Hessian_Diagonal=jnp.diag(self.Hessian_Network_Single(X,W))
		Gradient_TT=jnp.reshape(Hessian_Diagonal[0],(1,))
		Laplacian=jnp.reshape(jnp.sum(Hessian_Diagonal[1:]),(1,))
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

		Second_Order_Elements=self.Gradient_TT_And_Laplacian(X,W)
		Gradients_TT=Second_Order_Elements[0,:][None,:]
		Laplacians=Second_Order_Elements[1,:][None,:]
		return (Gradients_TT-(self.Data['C']**2)*Laplacians)


class Allen_Cahn_Scalar_Adaptive(Wrapper_Scalar_Adaptive):

	""" Allen-Cahn Scalar Problem Upon Adaptive PINN

		Input Vector -> [T;X]
		- T: Time Variable
		- X: Space Variable """


	def __init__(self,Architecture,Domain,Data,Adaptivity):
		super().__init__(Architecture,Domain,Data,Adaptivity)
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


class Helmholtz_Scalar_Adaptive(Wrapper_Scalar_Adaptive):

	""" Helmoltz Scalar Problem Upon Adaptive PINN

		Input Vector -> X: Space Variable """


	def __init__(self,Architecture,Domain,Data,Adaptivity):
		super().__init__(Architecture,Domain,Data,Adaptivity)
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
