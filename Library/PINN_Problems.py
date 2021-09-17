#! /usr/bin/python3

from PINN_Ground import *

class Poisson_Basic(PINN_Basic,Geometry_Basic):

	def __init__(self,ID,HL,NPL,Sigma,Domain,NResPts,NBouPts,BouLabs,SRC,EX_Bou):
		PINN_Basic.__init__(ID,1,HL,NPL,Sigma)
		Geometry_Basic.__init(Domain,NResPts,NBouPts,BouLabs)
		self.Source=SRC
		self.Exact_Boundary=EX_Bou
		self.Gradient_Network=jax.grad(self.Network)
		self.Hessian_Network=jax.jacobian(self.Gradient_Network)

	def Laplacian(self,X):

		""" Network Laplacian Computation """

		return jnp.sum(jnp.diag(self.Hessian_Network(X)))
	Laplacian=jax.jit(Laplacian)

	def PDE(self):

		""" PDE Loss Computation """

		return jnp.sum((self.Source(self.Residual_Points)-jnp.asarray(jax.vmap(self.Laplacian,in_axes=1,out_axes=1)(self.Residual_Points)))**2)/self.Number_Residuals
	PDE=jax.jit(PDE)

	def BC(self):

		""" Boundary Conditions Loss Computation """

		Result=0.0
		Dirichlet_Points=jnp.concatenate(self.Boundary_Lists[BouLabs=='Dirichlet'],axis=1)
		Neumann_Lists=[[self.Boundary_Lists[i],i] for i,bc in enumerate(BouLabs) if (bc=='Neumann')]
		Periodic_Lists=self.Boundary_Lists[BouLabs=='Periodic']
		Result+=jnp.sum((self.Exact_Boundary(Dirichlet_Points)-self.Network(Dirichlet_Points))**2)
		for FacePoints,FaceIndex in Neumann_Lists:
			Result+=jnp.sum((self.Exact_Boundary(FacePoints)-jax.vmap(jnp.inner,in_axes=(1,None),out_axes=1)(self.Gradient_Network(FacePoints),self.Boundary_Normals[:,FaceIndex]))**2)
		for l in range(len(Periodic_Lists)//2):
			Result+=jnp.sum((self.Network(Periodic_Lists[2*l])-self.Network(Periodic_Lists[2*l+1]))**2)
		return Result/self.Number_Boundary_Spots
	BC=jax.jit(BC)
