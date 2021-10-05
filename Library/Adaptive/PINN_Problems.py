#! /usr/bin/python3

from Adaptive.PINN_Wrappers import *


class Poisson_Scalar_Adaptive(Wrapper_Scalar_Adaptive):

	""" Poisson Scalar Problem Upon Adaptive PINN """


	def __init__(self,ID,Sigma,AdaFeatures,Domain,NResPts,NBouPts,BouLabs,SRC,Ex_Bou_D,Ex_Bou_N):
		super().__init__(ID,Sigma,AdaFeatures,Domain,NResPts,NBouPts,BouLabs,SRC,Ex_Bou_D,Ex_Bou_N)
		self.Equation=self.Laplacian


	@partial(jax.jit,static_argnums=(0))
	def SingleLaplacian(self,X,W,A,N):

		""" Single Input Contribution/Component Network Laplacian """

		return jnp.sum(jnp.diag(self.Hessian_Network_Single(X,W,A,N)))


	@partial(jax.jit,static_argnums=(0))
	def Laplacian(self,X,W,A,N):

		""" Overall Network Laplacian Computation """

		return (jax.vmap(self.SingleLaplacian,in_axes=(1,None,None,None))(X,W,A,N))[None,:]


class Burgers_Scalar_Adaptive(Wrapper_Scalar_Adaptive):

	""" Burgers Scalar Problem Upon Adaptive PINN

		Input Vector -> [T;X]
		- T: Time Variable
		- X: 1D Space Variable """


	def __init__(self,Sigma,AdaFeatures,Domain,NResPts,NBouPts,BouLabs,SRC,Ex_Bou_D,Ex_Bou_N,Nu):
		super().__init__(2,Sigma,AdaFeatures,Domain,NResPts,NBouPts,BouLabs,SRC,Ex_Bou_D,Ex_Bou_N)
		self.Nu=Nu
		self.Equation=self.Burgers_Left_Hand_Side


	@partial(jax.jit,static_argnums=(0))
	def SingleLaplacian(self,X,W,A,N):

		""" Single Input Contribution/Component Network Laplacian """

		return jnp.diag(self.Hessian_Network_Single(X,W,A,N))[1]


	@partial(jax.jit,static_argnums=(0))
	def Laplacian(self,X,W,A,N):

		""" Overall Network Laplacian Computation """

		return (jax.vmap(self.SingleLaplacian,in_axes=(1,None,None,None))(X,W,A,N))[None,:]


	@partial(jax.jit,static_argnums=(0))
	def Burgers_Left_Hand_Side(self,X,W,A,N):

		""" Overall Network Burgers Equation Left Hand Side Computation """

		Evaluations=self.Network_Multiple(X,W,A,N)
		Gradients=jax.vmap(self.Gradient_Network_Single,in_axes=(1,None,None,None),out_axes=1)(X,W,A,N)
		Gradients_T=Gradients[0,:][None,:]
		Gradients_X=Gradients[1,:][None,:]
		Laplacians=self.Laplacian(X,W,A,N)
		return (Gradients_T+Evaluations*Gradients_X-self.Nu*Laplacians)


class Heat_Scalar_Adaptive(Wrapper_Scalar_Adaptive):

	""" Heat Scalar Problem Upon Adaptive PINN

		Input Vector -> [T;X]
		- T: Time Variable
		- X: Space Variable """


	def __init__(self,ID,Sigma,AdaFeatures,Domain,NResPts,NBouPts,BouLabs,SRC,Ex_Bou_D,Ex_Bou_N,Nu):
		super().__init__(ID,Sigma,AdaFeatures,Domain,NResPts,NBouPts,BouLabs,SRC,Ex_Bou_D,Ex_Bou_N)
		self.Nu=Nu
		self.Equation=self.Heat_Left_Hand_Side


	@partial(jax.jit,static_argnums=(0))
	def SingleLaplacian(self,X,W,A,N):

		""" Single Input Contribution/Component Network Laplacian """

		return jnp.sum(jnp.diag(self.Hessian_Network_Single(X,W,A,N))[1:])


	@partial(jax.jit,static_argnums=(0))
	def Laplacian(self,X,W,A,N):

		""" Overall Network Laplacian Computation """

		return (jax.vmap(self.SingleLaplacian,in_axes=(1,None,None,None))(X,W,A,N))[None,:]


	@partial(jax.jit,static_argnums=(0))
	def Heat_Left_Hand_Side(self,X,W,A,N):

		""" Overall Network Heat Equation Left Hand Side Computation """

		Gradients=jax.vmap(self.Gradient_Network_Single,in_axes=(1,None,None,None),out_axes=1)(X,W,A,N)
		Gradients_T=Gradients[0,:][None,:]
		Laplacians=self.Laplacian(X,W,A,N)
		return (Gradients_T-self.Nu*Laplacians)


class Wave_Scalar_Adaptive(Wrapper_Scalar_Adaptive):

	""" Wave Scalar Problem Upon Adaptive PINN

		Input Vector -> [T;X]
		- T: Time Variable
		- X: Space Variable """


	def __init__(self,ID,Sigma,AdaFeatures,Domain,NResPts,NBouPts,BouLabs,SRC,Ex_Bou_D,Ex_Bou_N,C):
		super().__init__(ID,Sigma,AdaFeatures,Domain,NResPts,NBouPts,BouLabs,SRC,Ex_Bou_D,Ex_Bou_N)
		self.C=C
		self.Equation=self.Wave_Left_Hand_Side


	@partial(jax.jit,static_argnums=(0))
	def Single_Gradient_TT_And_Laplacian(self,X,W,A,N):

		""" Single Network Input Contribution/Component Of:
		 	- Second Time Derivative
			- Laplacian """

		Hessian_Diagonal=jnp.diag(self.Hessian_Network_Single(X,W,A,N))
		Gradient_TT=jnp.reshape(Hessian_Diagonal[0],(1,))
		Laplacian=jnp.reshape(jnp.sum(Hessian_Diagonal[1,:]),(1,))
		return jnp.concatenate((Gradient_TT,Laplacian),axis=0)


	@partial(jax.jit,static_argnums=(0))
	def Gradient_TT_And_Laplacian(self,X,W,A,N):

		""" Overall Network Computation Of:
		 	- Second Time Derivative
			- Laplacian """

		return (jax.vmap(self.Single_Gradient_TT_And_Laplacian,in_axes=(1,None,None,None),out_axes=1)(X,W,A,N))


	@partial(jax.jit,static_argnums=(0))
	def Wave_Left_Hand_Side(self,X,W,A,N):

		""" Overall Network Wave Equation Left Hand Side Computation """

		Second_Order_Elements=Gradient_TT_And_Laplacian(X,W,A,N)
		Gradients_TT=Second_Order_Elements[0,:][None,:]
		Laplacians=Second_Order_Elements[1,:][None,:]
		return (Gradients_TT-(self.C**2)*Laplacians)


class Allen_Cahn_Scalar_Adaptive(Wrapper_Scalar_Adaptive):

	""" Allen-Cahn Scalar Problem Upon Adaptive PINN

		Input Vector -> [T;X]
		- T: Time Variable
		- X: Space Variable """


	def __init__(self,ID,Sigma,AdaFeatures,Domain,NResPts,NBouPts,BouLabs,SRC,Ex_Bou_D,Ex_Bou_N,Gamma1,Gamma2):
		super().__init__(ID,Sigma,AdaFeatures,Domain,NResPts,NBouPts,BouLabs,SRC,Ex_Bou_D,Ex_Bou_N)
		self.Gamma1=Gamma1
		self.Gamma2=Gamma2
		self.Equation=self.Allen_Cahn_Left_Hand_Side


	@partial(jax.jit,static_argnums=(0))
	def SingleLaplacian(self,X,W,A,N):

		""" Single Input Contribution/Component Network Laplacian """

		return jnp.sum(jnp.diag(self.Hessian_Network_Single(X,W,A,N))[1:])


	@partial(jax.jit,static_argnums=(0))
	def Laplacian(self,X,W,A,N):

		""" Overall Network Laplacian Computation """

		return (jax.vmap(self.SingleLaplacian,in_axes=(1,None,None,None))(X,W,A,N))[None,:]


	@partial(jax.jit,static_argnums=(0))
	def Allen_Cahn_Left_Hand_Side(self,X,W,A,N):

		""" Overall Network Allen-Cahn Equation Left Hand Side Computation """

		Evaluations=self.Network_Multiple(X,W,A,N)
		Gradients=jax.vmap(self.Gradient_Network_Single,in_axes=(1,None,None,None),out_axes=1)(X,W,A,N)
		Gradients_T=Gradients[0,:][None,:]
		Laplacians=self.Laplacian(X,W,A,N)
		return (Gradients_T-self.Gamma1*Laplacians-self.Gamma2*(Evaluations-Evaluations**3))


class Helmholtz_Scalar_Adaptive(Wrapper_Scalar_Adaptive):

	""" Helmoltz Scalar Problem Upon Adaptive PINN

		Input Vector -> X: Space Variable """


	def __init__(self,ID,Sigma,AdaFeatures,Domain,NResPts,NBouPts,BouLabs,SRC,Ex_Bou_D,Ex_Bou_N,K):
		super().__init__(ID,Sigma,AdaFeatures,Domain,NResPts,NBouPts,BouLabs,SRC,Ex_Bou_D,Ex_Bou_N)
		self.K=K
		self.Equation=self.Helmholtz_Left_Hand_Side


	@partial(jax.jit,static_argnums=(0))
	def SingleLaplacian(self,X,W,A,N):

		""" Single Input Contribution/Component Network Laplacian """

		return jnp.sum(jnp.diag(self.Hessian_Network_Single(X,W,A,N)))


	@partial(jax.jit,static_argnums=(0))
	def Laplacian(self,X,W,A,N):

		""" Overall Network Laplacian Computation """

		return (jax.vmap(self.SingleLaplacian,in_axes=(1,None,None,None))(X,W,A,N))[None,:]


	@partial(jax.jit,static_argnums=(0))
	def Helmholtz_Left_Hand_Side(self,X,W,A,N):

		""" Overall Network Helmholtz Equation Left Hand Side Computation """

		Evaluations=self.Network_Multiple(X,W,A,N)
		Laplacians=self.Laplacian(X,W,A,N)
		return (Laplacians+(self.K**2)*Evaluations)
