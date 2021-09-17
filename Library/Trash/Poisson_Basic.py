#! /usr/bin/python3

from PINN_Ground import *

class Poisson_Basic(PINN_Basic):

	""" POISSON PROBLEM

		Problem Features Embedded:

		PDE:
		- Residual PDE Equation
		- Boundary Condition Equation

		Geometry:
		- Residual Points
		- Boundary Points """

	def __init__(self,ID,HL,NPL,Sigma,Domain,NResPts,NBouPts,BouLabs,SRC,EX_Bou):
		super().__init__(ID,1,HL,NPL,Sigma)
		self.Domain=Domain
		self.Number_Residuals=NResPts
		self.Number_Boundary_Spots=NBouPts
		self.Residual_Points=Uts.Sample_Interior(Domain,NResPts)
		self.Boundary_Lists=Uts.Sample_Boundary(Domain,NBouPts)
		self.Boundary_Labels=BouLabs
		self.Source=SRC
		self.Exact_Boundary=EX_Bou
		self.Gradient_Network=jax.grad(self.Network)
		self.Hessian_Network=jax.jacobian(self.Gradient_Network)

	def Laplacian(self,X):
		return jnp.sum(jnp.diag(self.Hessian_Network(X)))
	Laplacian=jax.jit(Laplacian)

	def PDE(self):
		return jnp.sum((self.Source(self.Residual_Points)-jnp.asarray(jax.vmap(self.Laplacian,in_axes=1,out_axes=1)(self.Residual_Points)))**2)/self.Number_Residuals
	PDE=jax.jit(PDE)

	def Normal(self,X):
		Vector=np.zeros((self.Input_Dimension,1))
		for n in range(self.Input_Dimension):
			Add_First=np.zeros()
			Add_Second=

	def BC(self):
		Result=0.0
		Dirichlet_Points=jnp.concatenate(self.Boundary_Lists[BouLabs=='Dirichlet'],axis=0).T
		Neumann_Points=jnp.concatenate(self.Boundary_Lists[BouLabs=='Neumann'],axis=0).T
		Periodic_Lists=self.Boundary_Lists[BouLabs=='Periodic']
		for x in Dirichlet_Points:
			Result+=(self.Exact_Boundary(x)-self.Network(x))**2
		for x in Neumann_Points:
			Result+=(self.Exact_Boundary(x)-jnp.inner(self.Gradient_Network(x),self.Normal(x)))**2
		for l in range(len(Periodic_Lists)//2):
			N=Periodic_Lists[2*l].shape[1]
			Result+=jnp.sum((self.Network(Periodic_Lists[2*l])-self.Network(Periodic_Lists[2*l+1]))**2)
		return Result/self.Number_Boundary_Spots
	BC=jax.jit(BC)
