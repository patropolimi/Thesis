#! /usr/bin/python3

from PINN_Utilities import *


class PINN_Basic:

	""" Standard Physics-Informed Neural Network """


	def __init__(self,ID,OD,HL,NPL,Sigma):
		self.Weights=Glorot_Basic(ID,OD,HL,NPL)
		self.Activation=Sigma


	@partial(jax.jit,static_argnums=(0))
	def Network(self,X,W=None):

		""" Network Application """

		if W is None:
			W=self.Weights

		Y=X
		for l in range(len(W)-1):
			Y=self.Activation(W[l][:,:-1]@Y+W[l][:,-1:])
		return W[-1][:,:-1]@Y+W[-1][:,-1:]


class Geometry_Basic:

	""" Standard Hyper-Rectangular Geometry For Physics-Informed Neural Networks

		Content:
		- Domain Coordinates
		- Residual Points
		- Boundary Labels, Points & Normal Vectors """


	def __init__(self,Domain,NResPts,NBouPts,BouLabs):
		self.Domain=Domain
		self.Residual_Points=Sample_Interior(Domain,NResPts)
		self.Boundary_Lists=Sample_Boundary(Domain,NBouPts)
		self.Number_Residuals=NResPts
		self.Number_Boundary_Spots=sum([self.Boundary_Lists[i].shape[1] for i in range(len(Boundary_Lists))])
		self.Boundary_Normals=Set_Normals(Domain.shape[0])
		self.Boundary_Labels=BouLabs
