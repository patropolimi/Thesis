#! /usr/bin/python3

from PINN_Ground import *

class Poisson_Basic(PINN_Basic,Geometry_Basic):

	""" Poisson Problem Wrapper """


	def __init__(self,ID,HL,NPL,Sigma,Domain,NResPts,NBouPts,BouLabs,SRC,EX_Bou):
		PINN_Basic.__init__(self,ID,1,HL,NPL,Sigma)
		Geometry_Basic.__init(self,Domain,NResPts,NBouPts,BouLabs)
		self.Source=SRC
		self.Exact_Boundary=EX_Bou
		self.Gradient_Network=jax.jit(jax.grad(self.Network))
		self.Hessian_Network=jax.jit(jax.jacobian(self.Gradient_Network))
		self.Gradient_Cost=jax.jit(jax.grad(self.Cost))


	@partial(jax.jit,static_argnums=(0))
	def Laplacian(self,X=self.Residual_Points,W=self.Weights):

		""" Network Laplacian Computation """

		return jnp.sum(jnp.diag(self.Hessian_Network(X,W)))


	@partial(jax.jit,static_argnums=(0))
	def PDE(self,X=self.Residual_Points,W=self.Weights):

		""" PDE Loss Computation """

		return jnp.sum((self.Source(X)-jnp.asarray(jax.vmap(self.Laplacian,in_axes=1,out_axes=1)(X,W)))**2)/self.Number_Residuals


	@partial(jax.jit,static_argnums=(0))
	def BC(self,X=self.Boundary_Lists,W=self.Weights):

		""" Boundary Conditions Loss Computation """

		Result=0.0
		Dirichlet_Points=jnp.concatenate(X[self.BouLabs=='Dirichlet'],axis=1)
		Neumann_Lists=[[X[i],i] for i,bc in enumerate(self.BouLabs) if (bc=='Neumann')]
		Periodic_Lists=X[self.BouLabs=='Periodic']
		Result+=jnp.sum((self.Exact_Boundary(Dirichlet_Points)-self.Network(Dirichlet_Points,W))**2)
		for FacePoints,FaceIndex in Neumann_Lists:
			Result+=jnp.sum((self.Exact_Boundary(FacePoints)-jax.vmap(jnp.inner,in_axes=(1,None),out_axes=1)(self.Gradient_Network(FacePoints,W),self.Boundary_Normals[:,FaceIndex]))**2)
		for l in range(len(Periodic_Lists)//2):
			Result+=jnp.sum((self.Network(Periodic_Lists[2*l],W)-self.Network(Periodic_Lists[2*l+1],W))**2)
		return Result/self.Number_Boundary_Spots

	@partial(jax.jit,static_argnums=(0))
	def Cost(self,W=self.Weights,XR=self.Residual_Points,XB=self.Boundary_Lists):

		""" Cost Function Computation """

		return self.PDE(XR,W)+self.BC(XB,W)
