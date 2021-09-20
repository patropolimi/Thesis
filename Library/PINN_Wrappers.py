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
		self.Dirichlet_Points,self.Dirichlet_Values,self.Neumann_Lists,self.Neumann_Values,self.Periodic_Lists=Set_Boundary_Points_And_Values(self.Boundary_Lists,BouLabs,Ex_Bou_D,Ex_Bou_N)
		self.Gradient_Network_Single=jax.jit(jax.grad(self.Network_Single))
		self.Hessian_Network_Single=jax.jit(jax.jacobian(self.Gradient_Network_Single))
		self.Gradient_Cost=jax.jit(jax.grad(self.Cost))
		self.Equation=None


	@partial(jax.jit,static_argnums=(0))
	def Network_Single(self,X,W=None):

		""" Network Application To Single Input X -> Scalar Output

			Requirement:
			- X: 1-Dimensional Array """

		if W is None:
			W=self.Weights

		Y=X
		for l in range(len(W)-1):
			Y=self.Activation(W[l][:,:-1]@Y+W[l][:,-1])
		return jnp.sum(W[-1][:,:-1]@Y+W[-1][:,-1])


	@partial(jax.jit,static_argnums=(0))
	def PDE(self,X=None,W=None):

		""" PDE Loss Computation

			Legend:
			- X[0] -> Residual_Points
			- X[1] -> Number Of Residual Points """

		if X is None:
			X=[self.Residual_Points,self.Number_Residuals]

		return jnp.sum((self.Source(X[0])-self.Equation(X[0],W))**2)/X[1]


	@partial(jax.jit,static_argnums=(0))
	def BC(self,X=None,W=None):

		""" Boundary Conditions Loss Computation

			Legend:
			- X[0] -> Dirichlet_Points
			- X[1] -> Dirichlet_Values
			- X[2] -> Neumann_Lists
			- X[3] -> Neumann_Values
			- X[4] -> Periodic_Lists
			- X[5] -> Number Of Boundary Spots """

		if X is None:
			X=[self.Dirichlet_Points,self.Dirichlet_Values,self.Neumann_Lists,self.Neumann_Values,self.Periodic_Lists,self.Number_Boundary_Spots]

		Result=jnp.sum((X[1]-self.Network_Multiple(X[0],W))**2)
		for i,[FacePoints,FaceIndex] in enumerate(X[2]):
			FaceGradients=jax.vmap(self.Gradient_Network_Single,in_axes=(1,None),out_axes=1)(FacePoints,W)
			Result+=jnp.sum((X[3][i]-(jax.vmap(jnp.inner,in_axes=(1,None))(FaceGradients,self.Boundary_Normals[:,FaceIndex]))[None,:])**2)
		for l in range(len(X[4])//2):
			Result+=jnp.sum((self.Network_Multiple(X[4][2*l],W)-self.Network_Multiple(X[4][2*l+1],W))**2)
		return Result/X[5]


	@partial(jax.jit,static_argnums=(0))
	def Cost(self,W=None,XR=None,XB=None):

		""" Cost Function Computation

			Derivative Requirement:
			- Gradient_Cost Needs W To Be Explicitely Passed """

		return self.PDE(XR,W)+self.BC(XB,W)
