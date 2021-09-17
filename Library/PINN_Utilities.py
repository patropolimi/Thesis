#! /usr/bin/python3

import jax
import jax.numpy as jnp
import numpy as np
import smt.sampling_methods as sm

Generator=np.random.default_rng(seed=time.time_ns())

def Random_Seed():

	""" Generating Random Seed """

	return Generator.integers(0,1e10)

def Glorot_Basic(Input_Dimension,Output_Dimension,Hidden_Layers,Neurons_Per_Layer):

	""" Standard Glorot Initialization """

	np.random.seed(Random_Seed())
	Layers=[Input_Dimension]+[Neurons_Per_Layer]*Hidden_Layers+[Output_Dimension]
	Weights=[]
	for l in range(Hidden_Layers+1):
		Weights.append(np.concatenate((np.random.randn(Layers[l+1],Layers[l])*np.sqrt(2/(Layers[l+1]+Layers[l])),np.zeros((Layers[l+1],1))),axis=1))
	return Weights

def Sample_Interior(Domain,N,Dist=1e-3):

	""" Uniform Sampling (N Spots) Of Domain Interior """

	Internal=Domain.copy()
	Internal[:,0]+=Dist/2
	Internal[:,1]-=Dist/2
	return sm.FullFactorial(xlimits=Internal)(N)

def Sample_Boundary(Domain,N,Dist=1e-3):

	""" Uniform Sampling Of Domain Boundary

		N -> List(List(2 Integers)):
		- N[d][0] -> Points On First Boundary Along d-th Direction/Dimension
		- N[d][1] -> Points On Second Boundary Along d-th Direction/Dimension

		Boundary_Points -> List Of 2*Problem_Input_Dimension Elements:
		- Boundary_Points[2*d] -> Array Of Boundary Points (Columnwise) On Lower Bound Hyper-Face Of d-th Direction/Dimension
		- Boundary_Poinyts[2*d+1] -> Array Of Boundary Points (Columnwise) On Upper Bound Hyper-Face Of d-th Direction/Dimension """

	Dim=len(N)
	Boundary_Points=[]
	Internal=Domain.copy()
	Internal[:,0]+=Dist/2
	Internal[:,1]-=Dist/2
	for d in range(Dim):
		Sampling=sm.FullFactorial(xlimits=np.concatenate((Internal[:d,:],Internal[d+1,:]),axis=0))
		Samples_LB=Sampling(N[d][0])
		Samples_UB=Sampling(N[d][1])
		Boundary_Points.append(np.concatenate((Samples_LB[:,:d],Domain[d,0]*np.ones((N[d][0],1)),Samples_LB[:,d+1:]),axis=1).T)
		Boundary_Points.append(np.concatenate((Samples_UB[:,:d],Domain[d,1]*np.ones((N[d][1],1)),Samples_UB[:,d+1:]),axis=1).T)
	return Boundary_Points

def Set_Normals(Dim):

	""" Setting Normal Vectors For Dim-Dimensional Hyper-Rectangular Domain """

	Normals=np.zeros((Dim,2*Dim))
	Normals[:,:1:]=-np.eye(Dim)
	Normals[:,1:1:]=np.eye(Dim)
	return Normals
