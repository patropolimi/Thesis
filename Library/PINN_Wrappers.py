#! /usr/bin/python3

from PINN_Grounds import *


class Wrapper_Scalar_Basic(PINN_Basic,Geometry_HyperRectangular):

	""" Wrapper For Scalar-Output Problems Built Upon Basic PINN """


	def __init__(self,ID,HL,NPL,Sigma,Domain,NResPts,NBouPts,BouLabs,SRC,Ex_Bou_D,Ex_Bou_N):
		PINN_Basic.__init__(self,ID,1,HL,NPL,Sigma)
		Geometry_HyperRectangular.__init__(self,Domain,NResPts,NBouPts,BouLabs)
		self.Source=jax.jit(SRC)
		self.Exact_Boundary_Dirichlet=jax.jit(Ex_Bou_D)
		self.Exact_Boundary_Neumann=jax.jit(Ex_Bou_N)
		self.Residual_Values=self.Source(self.Residual_Points)
		self.Dirichlet_Lists,self.Dirichlet_Values,self.Neumann_Lists,self.Neumann_Values,self.Periodic_Lists,self.Periodic_Lower_Points,self.Periodic_Upper_Points=Set_Boundary_Points_And_Values(self.Boundary_Lists,BouLabs,Ex_Bou_D,Ex_Bou_N)
		self.Gradient_Network_Single=jax.jit(jax.grad(self.Network_Single))
		self.Hessian_Network_Single=jax.jit(jax.jacobian(self.Gradient_Network_Single))
		self.Gradient_Cost=jax.jit(jax.grad(self.Cost))
		self.Equation=None


	@partial(jax.jit,static_argnums=(0))
	def Network_Single(self,X,W):

		""" Network Application To Single Input X

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

		return {'Dirichlet_Lists': self.Dirichlet_Lists,'Dirichlet_Values': self.Dirichlet_Values,'Neumann_Lists': self.Neumann_Lists,'Neumann_Values': self.Neumann_Values,'Periodic_Lower_Points': self.Periodic_Lower_Points,'Periodic_Upper_Points': self.Periodic_Upper_Points,'Number_Boundary_Spots': self.Number_Boundary_Spots}


	@partial(jax.jit,static_argnums=(0))
	def BC(self,X,W):

		""" Boundary Conditions Loss Computation """

		Result=0.0
		for FacePoints,FaceValues in zip(X['Dirichlet_Lists'],X['Dirichlet_Values']):
			Result+=jnp.sum((FaceValues-self.Network_Multiple(FacePoints,W))**2)
		for [FacePoints,FaceIndex],FaceValues in zip(X['Neumann_Lists'],X['Neumann_Values']):
			FaceGradients=jax.vmap(self.Gradient_Network_Single,in_axes=(1,None),out_axes=1)(FacePoints,W)
			Result+=jnp.sum((FaceValues-(jax.vmap(jnp.inner,in_axes=(1,None))(FaceGradients,jnp.take(self.Boundary_Normals,FaceIndex,axis=1)))[None,:])**2)
		for LowerPoints,UpperPoints in zip(X['Periodic_Lower_Points'],X['Periodic_Upper_Points']):
			Result+=jnp.sum((self.Network_Multiple(LowerPoints,W)-self.Network_Multiple(UpperPoints,W))**2)
		return Result/X['Number_Boundary_Spots']


	@partial(jax.jit,static_argnums=(0))
	def Cost(self,W,XR,XB):

		""" Cost Function Computation """

		return self.PDE(XR,W)+self.BC(XB,W)


	def Plot_1D(self,X,W=None):

		""" Plot Network On X

			Requirements:
			- 1D-Input Network
			- X -> 1D Array """

		if W is None:
			W=self.Weights

		plt.plot(X,self.Network_Multiple(X[None,:],W)[0,:])
		plt.show()


	def Plot_2D(self,X,Y,W=None):

		""" Plot Network On Meshgrid X x Y

			Requirements:
			- 2D-Input Network
			- X,Y -> 1D Arrays """

		if W is None:
			W=self.Weights

		XG,YG=np.meshgrid(X,Y)
		X_Vals=X[None,:]
		NX=X_Vals.shape[1]
		Pts=[]
		for i in range(YG.shape[0]):
			Y_Vals=np.array(NX*[YG[i,0]])[None,:]
			Pts+=[np.concatenate((X_Vals,Y_Vals),axis=0)]
		Pts=np.concatenate(Pts,axis=1)
		Net_Values=self.Network_Multiple(Pts,W).reshape(XG.shape)
		Figure=plt.figure()
		Ax=plt.axes(projection='3d')
		Ax.plot_surface(XG,YG,Net_Values)
		plt.show()


	def Print_Cost(self):

		""" Prints Current Cost Function Value """

		print("Current Cost: %.3e " %(self.Cost(self.Weights,self.PDE_Default_X(),self.BC_Default_X())))


class Wrapper_Scalar_Adaptive(PINN_Adaptive,Geometry_HyperRectangular):

	""" Wrapper For Scalar-Output Problems Built Upon Adaptive PINN """


	def __init__(self,ID,HL,NPL,Sigma,N,Domain,NResPts,NBouPts,BouLabs,SRC,Ex_Bou_D,Ex_Bou_N,AdaFeatures):
		PINN_Adaptive.__init__(self,ID,1,HL,NPL,Sigma,N)
		Geometry_HyperRectangular.__init__(self,Domain,NResPts,NBouPts,BouLabs)
		self.Adaptive_Features=AdaFeatures
		self.L=[1.0,1.0]
		self.Source=jax.jit(SRC)
		self.Exact_Boundary_Dirichlet=jax.jit(Ex_Bou_D)
		self.Exact_Boundary_Neumann=jax.jit(Ex_Bou_N)
		self.Residual_Values=self.Source(self.Residual_Points)
		self.Dirichlet_Lists,self.Dirichlet_Values,self.Neumann_Lists,self.Neumann_Values,self.Periodic_Lists,self.Periodic_Lower_Points,self.Periodic_Upper_Points=Set_Boundary_Points_And_Values(self.Boundary_Lists,BouLabs,Ex_Bou_D,Ex_Bou_N)
		self.Gradient_Network_Single=jax.jit(jax.grad(self.Network_Single))
		self.Hessian_Network_Single=jax.jit(jax.jacobian(self.Gradient_Network_Single))
		self.Gradient_Cost=jax.jit(jax.grad(self.Cost))
		self.Equation=None


	@partial(jax.jit,static_argnums=(0))
	def Network_Single(self,X,W,A,N):

		""" Network Application To Single Input X

			Requirement:
			- X: 1-Dimensional Array """

		Y=X
		for l in range(len(W)-1):
			Y=self.Activation(N[l]*A[l]*(W[l][:,:-1]@Y+W[l][:,-1]))
		return jnp.sum(W[-1][:,:-1]@Y+W[-1][:,-1])


	def PDE_Default_X(self):

		""" Helper -> Provides PDE Default Argument X """

		return {'Residual_Points': self.Residual_Points, 'Number_Residuals': self.Number_Residuals}


	@partial(jax.jit,static_argnums=(0))
	def PDE(self,X,W,A,N):

		""" PDE Loss Computation """

		return jnp.sum((self.Source(X['Residual_Points'])-self.Equation(X['Residual_Points'],W,A,N))**2)/X['Number_Residuals']


	def BC_Default_X(self):

		""" Helper -> Provides PDE Default Argument X """

		return {'Dirichlet_Lists': self.Dirichlet_Lists,'Dirichlet_Values': self.Dirichlet_Values,'Neumann_Lists': self.Neumann_Lists,'Neumann_Values': self.Neumann_Values,'Periodic_Lower_Points': self.Periodic_Lower_Points,'Periodic_Upper_Points': self.Periodic_Upper_Points,'Number_Boundary_Spots': self.Number_Boundary_Spots}


	@partial(jax.jit,static_argnums=(0))
	def BC(self,X,W,A,N):

		""" Boundary Conditions Loss Computation """

		Result=0.0
		for FacePoints,FaceValues in zip(X['Dirichlet_Lists'],X['Dirichlet_Values']):
			Result+=jnp.sum((FaceValues-self.Network_Multiple(FacePoints,W,A,N))**2)
		for [FacePoints,FaceIndex],FaceValues in zip(X['Neumann_Lists'],X['Neumann_Values']):
			FaceGradients=jax.vmap(self.Gradient_Network_Single,in_axes=(1,None,None,None),out_axes=1)(FacePoints,W,A,N)
			Result+=jnp.sum((FaceValues-(jax.vmap(jnp.inner,in_axes=(1,None))(FaceGradients,jnp.take(self.Boundary_Normals,FaceIndex,axis=1)))[None,:])**2)
		for LowerPoints,UpperPoints in zip(X['Periodic_Lower_Points'],X['Periodic_Upper_Points']):
			Result+=jnp.sum((self.Network_Multiple(LowerPoints,W,A,N)-self.Network_Multiple(UpperPoints,W,A,N))**2)
		return Result/X['Number_Boundary_Spots']


	@partial(jax.jit,static_argnums=(0))
	def Cost(self,W,A,N,L,XR,XB):

		""" Cost Function Computation """

		return L[0]*self.PDE(XR,W,A,N)+L[1]*self.BC(XB,W,A,N)


	def Plot_1D(self,X,W=None,A=None,N=None):

		""" Plot Network On X

			Requirements:
			- 1D-Input Network
			- X -> 1D Array """

		if W is None:
			W=self.Weights
		if A is None:
			A=self.A
		if N is None:
			N=self.N

		plt.plot(X,self.Network_Multiple(X[None,:],W,A,N)[0,:])
		plt.show()


	def Plot_2D(self,X,Y,W=None,A=None,N=None):

		""" Plot Network On Meshgrid X x Y

			Requirements:
			- 2D-Input Network
			- X,Y -> 1D Arrays """

		if W is None:
			W=self.Weights
		if A is None:
			A=self.A
		if N is None:
			N=self.N

		XG,YG=np.meshgrid(X,Y)
		X_Vals=X[None,:]
		NX=X_Vals.shape[1]
		Pts=[]
		for i in range(YG.shape[0]):
			Y_Vals=np.array(NX*[YG[i,0]])[None,:]
			Pts+=[np.concatenate((X_Vals,Y_Vals),axis=0)]
		Pts=np.concatenate(Pts,axis=1)
		Net_Values=self.Network_Multiple(Pts,W,A,N).reshape(XG.shape)
		Figure=plt.figure()
		Ax=plt.axes(projection='3d')
		Ax.plot_surface(XG,YG,Net_Values)
		plt.show()


	def Print_Cost(self):

		""" Prints Current Cost Function Value """

		print("Current Cost: %.3e " %(self.Cost(self.Weights,self.A,self.N,self.L,self.PDE_Default_X(),self.BC_Default_X())))
