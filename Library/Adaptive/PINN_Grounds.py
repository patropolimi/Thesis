#! /usr/bin/python3

from PINN_Utilities import *


class PINN_Adaptive:

	""" Adaptive Version Physics-Informed Neural Network

		Techniques In Use:
		- Growing (GRW)
		- Residual Adaptive Refinement (RAR) """


	def __init__(self,Architecture,Adaptivity):
		self.Architecture=Architecture
		self.Adaptivity=Adaptivity
		self.Architecture['W'],self.Architecture['Rows'],self.Architecture['Cum']=Flatten_SetGlobals(eval(Architecture['Initialization'])(Architecture['Input_Dimension'],Architecture['Output_Dimension'],Architecture['Hidden_Layers'],Architecture['Neurons_Per_Layer']))


	@partial(jax.jit,static_argnums=(0))
	def Network_Multiple(self,X,W):

		""" Network Application To Multiple Inputs X (Columnwise) -> Columnwise Output

			Requirement:
			- X: 2-Dimensional Array """

		Y=X
		WL=ListMatrixize(W)
		for l in range(len(WL)-1):
			Y=jnp.concatenate((WL[l][:1,:-1]@Y+WL[l][:1,-1:],self.Architecture['Activation'](WL[l][1:,:-1]@Y+WL[l][1:,-1:])),axis=0)
		return (WL[-1][:,:-1]@Y+WL[-1][:,-1:])
