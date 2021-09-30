#! /usr/bin/python3

from PINN_Utilities import *


class PINN_Basic:

	""" Basic Version Physics-Informed Neural Network """


	def __init__(self,ID,OD,HL,NPL,Sigma):
		self.Weights=Glorot_Uniform(ID,OD,HL,NPL)
		self.Activation=Sigma


	@partial(jax.jit,static_argnums=(0))
	def Network_Multiple(self,X,W):

		""" Network Application To Multiple Inputs X (Columnwise) -> Columnwise Output

			Requirement:
			- X: 2-Dimensional Array """

		Y=X
		for l in range(len(W)-1):
			Y=self.Activation(W[l][:,:-1]@Y+W[l][:,-1:])
		return W[-1][:,:-1]@Y+W[-1][:,-1:]
