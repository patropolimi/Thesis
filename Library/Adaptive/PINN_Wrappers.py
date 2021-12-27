#! /usr/bin/python3

from Adaptive.PINN_Grounds import *


class Wrapper_Scalar_Adaptive(PINN_Adaptive,Geometry_HyperRectangular):

	""" Wrapper For Scalar-Output Problems Built Upon Adaptive PINN """


	def __init__(self,Architecture,Domain,Data,Adaptivity):
		Architecture['Output_Dimension']=1
		PINN_Adaptive.__init__(self,Architecture,Adaptivity)
		Geometry_HyperRectangular.__init__(self,Domain)
		self.Data=Data
		self.Data['Dirichlet_Lists'],self.Data['Dirichlet_Values'],self.Data['Neumann_Lists'],self.Data['Neumann_Values'],self.Data['Periodic_Lower_Lists'],self.Data['Periodic_Upper_Lists']=Set_Boundary_Points_And_Values(self.Domain['Boundary_Lists'],Domain['Boundary_Labels'],Data['Exact_Dirichlet'],Data['Exact_Neumann'])
		self.Gradient_Network_Single=jax.jit(jax.grad(self.Network_Single))
		self.Hessian_Network_Single=jax.jit(jax.jacobian(self.Gradient_Network_Single))
		self.Gradient_Cost=jax.jit(jax.grad(self.Cost))
		self.Value_And_Gradient_Cost=jax.jit(jax.value_and_grad(self.Cost))
		self.Pool_Residuals=Sample_Interior(Domain['Limits'],Adaptivity['Pool_Residuals_Size'])
		self.Equation=None


	@partial(jax.jit,static_argnums=(0))
	def Network_Single(self,X,W):

		""" Network Application To Single Input X

			Requirement:
			- X: 1-Dimensional Array """

		Y=X
		WL=ListMatrixize(W)
		for l in range(len(WL)-1):
			Y=jnp.concatenate((WL[l][:1,:-1]@Y+WL[l][:1,-1],self.Architecture['Activation'](WL[l][1:,:-1]@Y+WL[l][1:,-1])),axis=0)
		return jnp.sum(WL[-1][:,:-1]@Y+WL[-1][:,-1])


	def PDE_Default_X(self):

		""" Helper -> Provides PDE Default Argument X """

		return self.Domain['Residual_Points']


	@partial(jax.jit,static_argnums=(0))
	def PDE(self,X,W):

		""" PDE Loss Computation """

		return jnp.mean((self.Data['Source'](X)-self.Equation(X,W))**2)


	def BC_Default_X(self):

		""" Helper -> Provides BC Default Argument X """

		return {'Dirichlet_Lists': self.Data['Dirichlet_Lists'],'Dirichlet_Values': self.Data['Dirichlet_Values'],'Neumann_Lists': self.Data['Neumann_Lists'],'Neumann_Values': self.Data['Neumann_Values'],'Periodic_Lower_Lists': self.Data['Periodic_Lower_Lists'],'Periodic_Upper_Lists': self.Data['Periodic_Upper_Lists']}


	@partial(jax.jit,static_argnums=(0))
	def BC(self,X,W):

		""" Boundary Conditions Loss Computation """

		Result=0.0
		Count=0
		for FacePoints,FaceValues in zip(X['Dirichlet_Lists'],X['Dirichlet_Values']):
			Result+=jnp.sum((FaceValues-self.Network_Multiple(FacePoints,W))**2)
			Count+=FacePoints.shape[1]
		for [FacePoints,FaceIndex],FaceValues in zip(X['Neumann_Lists'],X['Neumann_Values']):
			FaceGradients=jax.vmap(self.Gradient_Network_Single,in_axes=(1,None),out_axes=1)(FacePoints,W)
			Result+=jnp.sum((FaceValues-(jax.vmap(jnp.inner,in_axes=(1,None))(FaceGradients,jnp.take(self.Domain['Boundary_Normals'],FaceIndex,axis=1)))[None,:])**2)
			Count+=FacePoints.shape[1]
		for [LowerPoints,FaceIndexLower],[UpperPoints,FaceIndexUpper] in zip(X['Periodic_Lower_Lists'],X['Periodic_Upper_Lists']):
			Result+=jnp.sum((self.Network_Multiple(LowerPoints,W)-self.Network_Multiple(UpperPoints,W))**2)
			FaceGradientsLower=jax.vmap(self.Gradient_Network_Single,in_axes=(1,None),out_axes=1)(LowerPoints,W)
			FaceGradientsUpper=jax.vmap(self.Gradient_Network_Single,in_axes=(1,None),out_axes=1)(UpperPoints,W)
			Result+=jnp.sum(((jax.vmap(jnp.inner,in_axes=(1,None))(FaceGradientsLower,jnp.take(self.Domain['Boundary_Normals'],FaceIndexLower,axis=1)))+(jax.vmap(jnp.inner,in_axes=(1,None))(FaceGradientsUpper,jnp.take(self.Domain['Boundary_Normals'],FaceIndexUpper,axis=1))))**2)
			Count+=2*LowerPoints.shape[1]
		return Result/Count


	@partial(jax.jit,static_argnums=(0))
	def Cost(self,W,XR,XB):

		""" Cost Function Computation """

		return self.PDE(XR,W)+self.BC(XB,W)


	def Growing(self):

		""" Perform Growing

			Algorithm:
			- If Possible -> Double Neurons In Last Hidden Layer Initializing Relative Weights
			- Otherwise:
				- If Possible -> Add Hidden Layer Composed Of Min_Neurons_Per_Layer (Adaptivity Parameter) Neurons & Keep The Information Of The Former Output Weights Within New First Neuron (Of Last Hidden Layer) Weights
				- Otherwise -> Maximal Architecture Reached """

		WL=[np.asarray(w) for w in ListMatrixize(self.Architecture['W'])]
		if (self.Architecture['Neurons_Per_Layer'][-1]<self.Adaptivity['Max_Neurons_Per_Layer']):
			if (self.Architecture['Hidden_Layers']==1):
				WL_Addition_LastButOne=eval(self.Architecture['Initialization'])(self.Architecture['Input_Dimension'],self.Architecture['Neurons_Per_Layer'][-1],0,[])
			else:
				WL_Addition_LastButOne=eval(self.Architecture['Initialization'])(self.Architecture['Neurons_Per_Layer'][-2],self.Architecture['Neurons_Per_Layer'][-1],0,[])
			WL_Addition_Last=eval(self.Architecture['Initialization'])(self.Architecture['Neurons_Per_Layer'][-1]-1,1,0,[])
			WL[-2]=np.concatenate((WL[-2],WL_Addition_LastButOne[0]),axis=0)
			WL[-1]=np.concatenate((WL[-1],WL_Addition_Last[0]),axis=1)
			self.Architecture['Neurons_Per_Layer'][-1]*=2
			self.Architecture['W'],self.Architecture['Rows'],self.Architecture['Cum']=Flatten_SetGlobals(WL)
			print('Doubling Neurons In Last Hidden Layer')
		elif (self.Architecture['Hidden_Layers']<self.Adaptivity['Max_Hidden_Layers']):
			WL_Previous_Output_Weights=WL.pop()
			WL+=eval(self.Architecture['Initialization'])(self.Architecture['Neurons_Per_Layer'][-1],1,1,[self.Adaptivity['Min_Neurons_Per_Layer']])
			self.Architecture['Neurons_Per_Layer']+=[self.Adaptivity['Min_Neurons_Per_Layer']]
			self.Architecture['Hidden_Layers']+=1
			WL[-2][:1,:]=WL_Previous_Output_Weights
			WL[-1][0,0]=1.0
			self.Architecture['W'],self.Architecture['Rows'],self.Architecture['Cum']=Flatten_SetGlobals(WL)
			print('Adding Hidden Layer Of %d Neurons' %(self.Adaptivity['Min_Neurons_Per_Layer']))
		else:
			print('Maximum Architecture Size Reached: Impossible To Grow Any Further')


	def Residual_Adaptive_Refinement(self,Values_Pool):

		""" Perform Residual Adaptive Refinement

			Requirement:
			- Values_Pool: 2-D Array Containing PDE Evaluation Over Pool_Residuals

			Algorithm:
			- If Possible -> Double Residual Points Adding Elements Of Values_Pool On Which PDE Residual Is Highest
			- Otherwise -> Maximum Number Of Residual Points Reached """

		if (self.Domain['Number_Residuals']<self.Adaptivity['Max_Number_Residuals']):
			Selected_Indexes=np.argsort(np.ravel(Values_Pool))[-self.Domain['Number_Residuals']:]
			self.Domain['Residual_Points']=np.concatenate((self.Domain['Residual_Points'],self.Pool_Residuals[:,Selected_Indexes]),axis=1)
			self.Domain['Number_Residuals']*=2
			print('Doubling Residuals')
		else:
			print('Maximum Number Of Residuals Reached: Impossible To Add Any Further')


	def Plot_1D(self,X,W=None,S=None):

		""" Plot Network On X

			Requirements:
			- 1D-Input Network
			- X -> 1D Array """

		if W is None:
			W=self.Architecture['W']

		if S is not None:
			plt.plot(X,S(X))
		plt.plot(X,self.Network_Multiple(X[None,:],W)[0,:])
		plt.show()


	def Plot_2D(self,X,Y,W=None,S=None):

		""" Plot Network On Meshgrid X x Y

			Requirements:
			- 2D-Input Network
			- X,Y -> 1D Arrays """

		if W is None:
			W=self.Architecture['W']

		XG,YG=np.meshgrid(X,Y)
		NX=X.shape[0]
		NY=Y.shape[0]
		XV=X[None,:]
		Pts=[]
		for i in range(NY):
			YV=np.array(NX*[YG[i,0]])[None,:]
			Pts+=[np.concatenate((XV,YV),axis=0)]
		Pts=np.concatenate(Pts,axis=1)
		Network_Values=self.Network_Multiple(Pts,W).reshape(XG.shape)
		Axes=plt.axes(projection='3d')
		if S is not None:
			Axes.plot_surface(XG,YG,S(Pts).reshape(XG.shape))
		Axes.plot_surface(XG,YG,Network_Values)
		plt.show()


	def Print_Cost(self):

		""" Prints Current Cost Function Value """

		print('Current Cost: %.3e ' %(self.Cost(self.Architecture['W'],self.Pool_Residuals,self.BC_Default_X())))
