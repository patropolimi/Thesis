#! /usr/bin/python3

from PINN_Utilities import *


class PINN_Adaptive:

	""" Adaptive Version Physics-Informed Neural Network """


	def __init__(self,ID,OD,HL,NPL,Sigma,N):
		self.Weights=Glorot_Uniform_Basic(ID,OD,HL,NPL)
		self.Activation=Sigma
		self.A=1.0/N
		self.N=N


	@partial(jax.jit,static_argnums=(0))
	def Network_Multiple(self,X,W,A,N):

		""" Network Application To Multiple Inputs X (Columnwise) -> Columnwise Output

			Requirement:
			- X: 2-Dimensional Array """

		Y=X
		for l in range(len(W)-1):
			Y=self.Activation(N[l]*A[l]*(W[l][:,:-1]@Y+W[l][:,-1:]))
		return W[-1][:,:-1]@Y+W[-1][:,-1:]
