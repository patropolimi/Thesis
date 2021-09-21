#! /usr/bin/python3

from PINN_Grounds import *


class Problem_Scalar_Basic(PINN_Basic,Geometry_Basic):

	""" Basic Scalar Problem Wrapper """


	def __init__(self,ID,HL,NPL,Sigma,Domain,NResPts,NBouPts,BouLabs,SRC,Ex_Bou_D,Ex_Bou_N):
		PINN_Basic.__init__(self,ID,1,HL,NPL,Sigma)
		Geometry_Basic.__init__(self,Domain,NResPts,NBouPts,BouLabs)
		self.Source=jax.jit(SRC)
		self.Exact_Boundary_Dirichlet=jax.jit(Ex_Bou_D)
		self.Exact_Boundary_Neumann=jax.jit(Ex_Bou_N)
		self.Residual_Values=self.Source(self.Residual_Points)
		self.Dirichlet_Points,self.Dirichlet_Values,self.Neumann_Lists,self.Neumann_Values,self.Periodic_Lists,self.Periodic_Lower_Points,self.Periodic_Upper_Points=Set_Boundary_Points_And_Values(self.Boundary_Lists,BouLabs,Ex_Bou_D,Ex_Bou_N)
		self.Gradient_Network_Single=jax.jit(jax.grad(self.Network_Single))
		self.Hessian_Network_Single=jax.jit(jax.jacobian(self.Gradient_Network_Single))
		self.Gradient_Cost=jax.jit(jax.grad(self.Cost))
		self.Equation=None


	@partial(jax.jit,static_argnums=(0))
	def Network_Single(self,X,W):

		""" Network Application To Single Input X -> Scalar Output

			Requirement:
			- X: 1-Dimensional Array """

		Y=X
		for l in range(len(W)-1):
			Y=self.Activation(W[l][:,:-1]@Y+W[l][:,-1])
		return jnp.sum(W[-1][:,:-1]@Y+W[-1][:,-1])


	def PDE_Default_X(self):

		""" Helper -> Provides PDE Default Argument X """

		return {'Residual_Points': self.Residual_Points, 'Number_Residuals': self.Number_Residuals}


	@partial(jax.jit,static_argnums=(0))
	def PDE(self,X,W):

		""" PDE Loss Computation """

		return jnp.sum((self.Source(X['Residual_Points'])-self.Equation(X['Residual_Points'],W))**2)/X['Number_Residuals']


	def BC_Default_X(self):

		""" Helper -> Provides PDE Default Argument X """

		return {'Dirichlet_Points': self.Dirichlet_Points,'Dirichlet_Values': self.Dirichlet_Values,'Neumann_Lists': self.Neumann_Lists,'Neumann_Values': self.Neumann_Values,'Periodic_Lower_Points': self.Periodic_Lower_Points,'Periodic_Upper_Points': self.Periodic_Upper_Points,'Number_Boundary_Spots': self.Number_Boundary_Spots}


	@partial(jax.jit,static_argnums=(0))
	def BC(self,X,W):

		""" Boundary Conditions Loss Computation """

		Result=jnp.sum((X['Dirichlet_Values']-self.Network_Multiple(X['Dirichlet_Points'],W))**2)
		Neumann_Network_Values=[]
		for FacePoints,FaceIndex in X['Neumann_Lists']:
			FaceGradients=jax.vmap(self.Gradient_Network_Single,in_axes=(1,None),out_axes=1)(FacePoints,W)
			Neumann_Network_Values+=[(jax.vmap(jnp.inner,in_axes=(1,None))(FaceGradients,jnp.take(self.Boundary_Normals,FaceIndex,axis=1)))[None,:]]
		Neumann_Network_Values=jnp.concatenate(Neumann_Network_Values,axis=1)
		Result+=jnp.sum((X['Neumann_Values']-Neumann_Network_Values)**2)
		Result+=jnp.sum((self.Network_Multiple(X['Periodic_Lower_Points'],W)-self.Network_Multiple(X['Periodic_Upper_Points'],W))**2)
		return Result/X['Number_Boundary_Spots']


	@partial(jax.jit,static_argnums=(0))
	def Cost(self,W,XR,XB):

		""" Cost Function Computation """

		return self.PDE(XR,W)+self.BC(XB,W)
