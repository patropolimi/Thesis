#! /usr/bin/python3

from PINN_Utilities import *


class PINN_Adaptive:

	""" Adaptive Version Physics-Informed Neural Network

		Adaptive Features:
		- RAR -> Residual Adaptive Refinement [Mandatory]
		- NAE -> Network Auto Enhancing [Mandatory]
		- SAM -> Soft Attention Mechanism [Optional]
		- AAF -> Adaptive Activation Function [Optional] """


	def __init__(self,ID,OD,Sigma,AdaFeatures):
		self.RAR_Options=AdaFeatures['RAR']
		self.NAE_Options=AdaFeatures['NAE']
		self.SAM_Options=AdaFeatures['SAM']
		self.AAF_Options=AdaFeatures['AAF']
		self.Activation=Sigma
		self.A=np.array([1.0])
		self.N=np.array([1.0])
		W_List=Glorot_Uniform_Basic(ID,OD,1,AdaFeatures['NAE']['Initial_Neurons'])
		M_List=[np.ones_like(w) for w in W_List]
		self.Weights_On=Flatten_And_Update(W_List)
		self.Mask=FastFlatten(M_List)
		self.Hidden_Layers=1


	@partial(jax.jit,static_argnums=(0))
	def Network_Multiple(self,X,W,A,N):

		""" Network Application To Multiple Inputs X (Columnwise) -> Columnwise Output

			Requirement:
			- X: 2-Dimensional Array
			- W: 1-Dimensional Array Of Active Weights """

		Y=X
		WL=self.ListFill(W)
		for l in range(len(WL)-1):
			Y=self.Activation(N[l]*A[l]*(WL[l][:,:-1]@Y+WL[l][:,-1:]))
		return WL[-1][:,:-1]@Y+WL[-1][:,-1:]


	@partial(jax.jit,static_argnums=(0))
	def ListFill(self,W_Active):

		""" Create List[2D Array] Inserting Active Weights W_Active According To PINN's Mask """

		W_Full=np.zeros_like(self.Mask)
		W_Full[self.Mask]=W_Active
		return ListMatrixize(W_Full)
