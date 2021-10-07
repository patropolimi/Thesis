#! /usr/bin/python3

from Adaptive.PINN_Grounds import *


class Wrapper_Scalar_Adaptive(PINN_Adaptive,Geometry_HyperRectangular):

	""" Wrapper For Scalar-Output Problems Built Upon Adaptive PINN """


	def __init__(self,ID,Sigma,AdaFeatures,Domain,NResPts,NBouPts,BouLabs,SRC,Ex_Bou_D,Ex_Bou_N):
		PINN_Adaptive.__init__(self,ID,1,Sigma,AdaFeatures)
		Geometry_HyperRectangular.__init__(self,Domain,NResPts,NBouPts,BouLabs)
		self.L=np.ones(self.Number_Boundary_Spots+self.Number_Residuals)
		self.Source=jax.jit(SRC)
		self.Exact_Boundary_Dirichlet=jax.jit(Ex_Bou_D)
		self.Exact_Boundary_Neumann=jax.jit(Ex_Bou_N)
		self.Dirichlet_Lists,self.Dirichlet_Values,self.Neumann_Lists,self.Neumann_Values,self.Periodic_Lists,self.Periodic_Lower_Points,self.Periodic_Upper_Points=Set_Boundary_Points_And_Values(self.Boundary_Lists,BouLabs,Ex_Bou_D,Ex_Bou_N)
		self.Gradient_Network_Single=jax.jit(jax.grad(self.Network_Single))
		self.Hessian_Network_Single=jax.jit(jax.jacobian(self.Gradient_Network_Single))
		self.Gradient_Cost_W=jax.jit(jax.grad(self.Cost,argnums=0))
		self.Gradient_Cost_A=jax.jit(jax.grad(self.Cost,argnums=1))
		self.Gradient_Cost_L=jax.jit(jax.grad(self.Cost,argnums=3))
		self.Equation=None


	@partial(jax.jit,static_argnums=(0))
	def Network_Single(self,X,W,A,N):

		""" Network Application To Single Input X

			Requirement:
			- X: 1-Dimensional Array
			- W: 1-Dimensional Array Of Active Weights
			- M: Mask List[2-D Array] For Active & Passive Weights """

		Y=X
		WL=self.ListFill(W)
		for l in range(len(WL)-1):
			Y=self.Activation(N[l]*A[l]*(WL[l][:,:-1]@Y+WL[l][:,-1]))
		return jnp.sum(WL[-1][:,:-1]@Y+WL[-1][:,-1])


	def PDE_Default_X(self):

		""" Helper -> Provides PDE Default Argument X """

		return {'Residual_Points': self.Residual_Points, 'Number_Residuals': self.Number_Residuals}


	@partial(jax.jit,static_argnums=(0))
	def PDE(self,X,W,A,N,L_PDE):

		""" PDE Loss Computation """

		return jnp.sum((L_PDE*(self.Source(X['Residual_Points'])-self.Equation(X['Residual_Points'],W,A,N)))**2)/X['Number_Residuals']


	def BC_Default_X(self):

		""" Helper -> Provides PDE Default Argument X """

		return {'Dirichlet_Lists': self.Dirichlet_Lists,'Dirichlet_Values': self.Dirichlet_Values,'Neumann_Lists': self.Neumann_Lists,'Neumann_Values': self.Neumann_Values,'Periodic_Lower_Points': self.Periodic_Lower_Points,'Periodic_Upper_Points': self.Periodic_Upper_Points,'Number_Boundary_Spots': self.Number_Boundary_Spots}


	@partial(jax.jit,static_argnums=(0))
	def BC(self,X,W,A,N,L_BC):

		""" Boundary Conditions Loss Computation """

		Result=0.0
		Count_L=0
		Count_U=0
		for FacePoints,FaceValues in zip(X['Dirichlet_Lists'],X['Dirichlet_Values']):
			Count_U+=FaceValues.shape[1]
			Result+=jnp.sum((jnp.take(L_BC,jnp.arange(Count_L,Count_U))*(FaceValues-self.Network_Multiple(FacePoints,W,A,N)))**2)
			Count_L=Count_U
		for [FacePoints,FaceIndex],FaceValues in zip(X['Neumann_Lists'],X['Neumann_Values']):
			Count_U+=FaceValues.shape[1]
			FaceGradients=jax.vmap(self.Gradient_Network_Single,in_axes=(1,None,None,None),out_axes=1)(FacePoints,W,A,N)
			Result+=jnp.sum((jnp.take(L_BC,jnp.arange(Count_L,Count_U))*(FaceValues-(jax.vmap(jnp.inner,in_axes=(1,None))(FaceGradients,jnp.take(self.Boundary_Normals,FaceIndex,axis=1)))[None,:]))**2)
			Count_L=Count_U
		for LowerPoints,UpperPoints in zip(X['Periodic_Lower_Points'],X['Periodic_Upper_Points']):
			Count_U+=LowerPoints.shape[1]
			Result+=jnp.sum((jnp.take(L_BC,jnp.arange(Count_L,Count_U))*(self.Network_Multiple(LowerPoints,W,A,N)-self.Network_Multiple(UpperPoints,W,A,N)))**2)
			Count_L=Count_U
		return Result/X['Number_Boundary_Spots']


	@partial(jax.jit,static_argnums=(0))
	def Cost(self,W,A,N,L,XR,XB):

		""" Cost Function Computation """

		return self.PDE(XR,W,A,N,jnp.take(L,jnp.arange(-XB['Number_Residuals'],0,1)))+self.BC(XB,W,A,N,jnp.take(L,jnp.arange(0,XB['Number_Boundary_Spots'],1)))


	def RAR(self):

		""" Residual Adaptive Refinement

			Two Possible Sampling Methods:
			- RND -> Random
			- LHS -> Latin Hypercube Sampling """

		if (self.RAR_Options['Mode']=='RND'):
			Sampling=sm.Random(xlimits=self.Domain)
		elif (self.RAR_Options['Mode']=='LHS'):
			Sampling=sm.LHS(xlimits=self.Domain)
		Pool=Sampling(self.RAR_Options['PoolSize']).T
		Values=((self.Source(Pool)-self.Equation(Pool,W,A,N))[0,:])**2
		Standard=self.PDE(self.PDE_Default_X(),self.Weights_On,self.A,self.N,np.ones(self.Number_Residuals))
		Candidate_Indexes=np.argpartition(Values,-self.RAR_Options['MaxAdd'])[-self.RAR_Options['MaxAdd']:]
		Candidate_Points=Pool[:,Candidate_Indexes]
		Candidate_Values=Values[Candidate_Indexes]
		Additional_Points=Candidate_Points[:,Candidate_Values>Standard]
		Number_NewPoints=Additional_Points.shape[1]
		self.Residual_Points=np.concatenate((self.Residual_Points,Additional_Points),axis=1)
		self.Number_Residuals+=Number_NewPoints
		self.L=np.concatenate((self.L,np.ones(Number_NewPoints)),axis=0)


	def Update_A(self):

		""" Update Activation Parameters """

		Grad_A=self.Gradient_Cost_A(self.Weights_On,self.A,self.N,self.L,self.PDE_Default_X(),self.BC_Default_X())
		self.A-=self.AAF_Options['Learning_Rate']*Grad_A


	def Update_L(self):

		""" Update Loss Weights """

		Grad_L=self.Gradient_Cost_L(self.Weights_On,self.A,self.N,self.L,self.PDE_Default_X(),self.BC_Default_X())
		self.L+=self.SAM_Options['Learning_Rate']*Grad_L


	def Pruning(self,SV):

		""" Pruning Step Execution """

		WL=self.ListFill(self.Weights_On)
		SVL=self.ListFill(SV)
		List_Length=len(SVL)
		Beta=[self.NAE_Options['Relaxation'][l]/(SVL[l].shape[1]) for l in range(List_Length)]
		Divisor=[np.repeat(np.sum(np.abs(SVL[l]),axis=1)[:,None],SVL[l].shape[1],axis=1) for l in range(List_Length)]
		LRSI=[np.asarray(np.divide(np.abs(SVL[l]),Divisor[l],out=np.zeros_like(SVL[l]),where=(np.abs(Divisor[l])>EpsMachine))) for l in range(List_Length)]
		MaskList_New=[(LRSI[l]>Beta[l]) for l in range(List_Length)]
		CutOff(MaskList_New)
		self.Mask=FastFlatten(MaskList_New)
		self.Weights_On=jnp.take(FastFlatten(WL),self.Mask)


	def Growing(self):

		""" Growing Step Execution """

		self.Hidden_Layers+=1
		self.N=np.concatenate((self.N,np.array([self.AAF_Options['Scaling_Increase_Factor']**(self.Hidden_Layers-1)])),axis=0)
		self.A=np.concatenate((self.A,np.array([1.0/self.N[-1]])),axis=0)
		WL=self.ListFill(self.Weights_On)
		ML=ListMatrixize(self.Mask)
		Neurons_Last_ButOne=WL[-1].shape[1]-1
		Neurons_Last=np.ceil(self.NAE_Options['Initial_Neurons']*(self.NAE_Options['Neuron_Increase_Factor']**(self.Hidden_Layers-1)))
		Out_Weights=WL.pop()
		Out_Mask=ML.pop()
		I=jnp.eye(Neurons_Last_ButOne)
		WL+=Glorot_Normal(Neurons_Last_ButOne,1,1,Neurons_Last)
		ML+=[np.ones_like(w) for w in WL[-2:]]
		WL[-2]=WL[-2].at[:Neurons_Last_ButOne,:Neurons_Last_ButOne].set(I)
		WL[-1]=WL[-1].at[:,:Neurons_Last_ButOne].set(Out_Weights)
		ML[-1]=ML[-1].at[:,:Neurons_Last_ButOne].set(Out_Mask)
		CutOff(ML)
		self.Mask=Flatten_And_Update(ML)
		self.Weights_On=jnp.take(FastFlatten(WL),self.Mask)


	def Plot_1D(self,X,W=None,A=None,N=None,Iters=None,SaveName=None):

		""" Plot Network On X

			Requirements:
			- 1D-Input Network
			- X -> 1D Array """

		if W is None:
			W=self.Weights_On
		if A is None:
			A=self.A
		if N is None:
			N=self.N

		Figure=plt.figure()
		plt.plot(X,self.Network_Multiple(X[None,:],W,A,N)[0,:])
		if (Iters is not(None)):
			plt.suptitile(str(Iters)+' Iterations')
		if (SaveName is not(None)):
			plt.savefig(SaveName+'.eps',format='eps')
		plt.show()


	def Plot_2D(self,X,Y,W=None,A=None,N=None,Iters=None,SaveName=None):

		""" Plot Network On Meshgrid X x Y

			Requirements:
			- 2D-Input Network
			- X,Y -> 1D Arrays """

		if W is None:
			W=self.Weights_On
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
		if (Iters is not(None)):
			plt.suptitile(str(Iters)+' Iterations')
		if (SaveName is not(None)):
			plt.savefig(SaveName+'.eps',format='eps')
		plt.show()


	def Print_Cost(self):

		""" Prints Current Cost Function Value """

		print("Current Cost: %.3e " %(self.Cost(self.Weights_On,self.A,self.N,self.L,self.PDE_Default_X(),self.BC_Default_X())))
