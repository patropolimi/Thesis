#! /usr/bin/python3

import PINN_Utilities as Uts


class PINN_Basic:

	""" Standard Physics-Informed Neural Network """


	def __init__(self,ID,OD,HL,NPL,Sigma):
		self.Weights=Uts.Glorot_Basic(ID,OD,HL,NPL)
		self.Activation=Sigma


	@partial(jax.jit,static_argnums=(0))
	def Network(self,X,W=self.Weights,Sigma=self.Activation):

		""" Network Application """

		Y=X
		for l in range(len(W)-1):
			Y=Sigma(W[l][:,:-1]@Y+W[l][:,-1:])
		return W[-1][:,:-1]@Y+W[-1][:,-1:]


class Geometry_Basic:

	""" Standard Hyper-Rectangular Geometry For Physics-Informed Neural Networks

		Content:
		- Domain Coordinates
		- Residual Points
		- Boundary Labels, Points & Normal Vectors """


	def __init__(self,Domain,NResPts,NBouPts,BouLabs):
		self.Domain=Domain
		self.Number_Residuals=NResPts
		self.Number_Boundary_Spots=NBouPts
		self.Residual_Points=Uts.Sample_Interior(Domain,NResPts)
		self.Boundary_Lists=Uts.Sample_Boundary(Domain,NBouPts)
		self.Boundary_Normals=Uts.Set_Normals(Domain.shape[0])
		self.Boundary_Labels=BouLabs
