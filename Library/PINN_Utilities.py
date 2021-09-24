#! /usr/bin/python3

import dill
import time
import jax
import jax.numpy as jnp
import numpy as np
import smt.sampling_methods as sm
import matplotlib.pyplot as plt
from functools import partial
from type_templating import Template,TemplateParameter


jax.config.update("jax_enable_x64",True)


Generator=np.random.default_rng(seed=time.time_ns())


def Random_Seed():

	""" Generating Random Seed """

	return Generator.integers(0,1e5)


def Glorot_Uniform_Basic(Input_Dimension,Output_Dimension,Hidden_Layers,Neurons_Per_Layer):

	""" Standard Glorot Uniform Initialization """

	np.random.seed(Random_Seed())
	Layers=[Input_Dimension]+[Neurons_Per_Layer]*Hidden_Layers+[Output_Dimension]
	Weights=[]
	for l in range(Hidden_Layers+1):
		Limit=np.sqrt(6/(Layers[l+1]+Layers[l]))
		Weights.append(np.concatenate((np.random.uniform(-Limit,Limit,(Layers[l+1],Layers[l])),np.zeros((Layers[l+1],1))),axis=1))
	return Weights


def Sample_Interior(Domain,N,Dist=1e-3):

	""" Uniform Sampling (N Spots) Of Domain Interior """

	Internal=Domain.copy()
	Internal[:,0]+=Dist/2
	Internal[:,1]-=Dist/2
	return sm.FullFactorial(xlimits=Internal)(N).T


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
	if (Dim==1):
		if (N[0][0]):
			Boundary_Points+=[np.array([[Domain[0,0]]])]
		else:
			Boundary_Points+=[np.array([[]])]
		if (N[0][1]):
			Boundary_Points+=[np.array([[Domain[0,1]]])]
		else:
			Boundary_Points+=[np.array([[]])]
	else:
		Internal=Domain.copy()
		Internal[:,0]+=Dist/2
		Internal[:,1]-=Dist/2
		for d in range(Dim):
			Sampling=sm.FullFactorial(xlimits=np.concatenate((Internal[:d,:],Internal[d+1:,:]),axis=0))
			if (N[d][0]):
				Samples_LB=Sampling(N[d][0])
				Boundary_Points.append(np.concatenate((Samples_LB[:,:d],Domain[d,0]*np.ones((N[d][0],1)),Samples_LB[:,d:]),axis=1).T)
			else:
				Boundary_Points.append(np.array([[]]))
			if (N[d][1]):
				Samples_UB=Sampling(N[d][1])
				Boundary_Points.append(np.concatenate((Samples_UB[:,:d],Domain[d,1]*np.ones((N[d][1],1)),Samples_UB[:,d:]),axis=1).T)
			else:
				Boundary_Points.append(np.array([[]]))
	return Boundary_Points


def Set_Normals(Dim):

	""" Setting Normal Vectors For Dim-Dimensional Hyper-Rectangular Domain """

	Normals=np.zeros((Dim,2*Dim))
	Normals[:,::2]=-np.eye(Dim)
	Normals[:,1::2]=np.eye(Dim)
	return Normals


def Set_Boundary_Points_And_Values(BouLists,BouLabs,Ex_Bou_D,Ex_Bou_N):

	""" Setting Boundary Points & Values

	 	Output:
		- Dirichlet_Lists -> List[Columnwise Array Of Dirichlet Points]
		- Dirichlet_Values -> List[Columnwise Array Of Dirichlet Values]
		- Neumann_Lists -> List[[Columnwise Array Of Neumann Points On i-th Face,i]]
		- Neumann_Values -> List[Columnwise Array Of Neumann Values]
		- Periodic_Lists -> List[Columnwise Array Of Periodic Points]
		- Periodic_Lower_Points -> List[Columnwise Array Of Lower Periodic Points]
		- Periodic_Upper_Points -> List[Columnwise Array Of Upper Periodic Points] """

	NBL=len(BouLists)
	Dirichlet_Lists=[BouLists[i] for i in range(NBL) if (BouLabs[i]=='Dirichlet')]
	if Dirichlet_Lists:
		Dirichlet_Values=[Ex_Bou_D(FacePoints) for FacePoints in Dirichlet_Lists]
	else:
		Dirichlet_Values=[]
	Neumann_Lists=[[BouLists[i],i] for i,bc in enumerate(BouLabs) if (bc=='Neumann')]
	if Neumann_Lists:
		Neumann_Values=[Ex_Bou_N(FacePoints) for FacePoints,_ in Neumann_Lists]
	else:
		Neumann_Values=[]
	Periodic_Lists=[BouLists[i] for i in range(NBL) if (BouLabs[i]=='Periodic')]
	if Periodic_Lists:
		Periodic_Lower_Points=Periodic_Lists[::2]
		Periodic_Upper_Points=Periodic_Lists[1::2]
	else:
		Periodic_Lower_Points=[]
		Periodic_Upper_Points=[]
	return Dirichlet_Lists,Dirichlet_Values,Neumann_Lists,Neumann_Values,Periodic_Lists,Periodic_Lower_Points,Periodic_Upper_Points


def Flatten(ArrayList):

	""" Flatten ArrayList """

	global Rows,Cum

	L=len(ArrayList)
	Rows=np.zeros((L),dtype=int)
	Cum=np.zeros((L+1),dtype=int)
	ArrayFlat=[]
	for l in range(L):
		s=np.shape(ArrayList[l])
		Rows[l]=s[0]
		Cum[l+1]=s[0]*s[1]+Cum[l]
		ArrayFlat+=[np.ravel(ArrayList[l])]
	return np.asarray(np.concatenate(ArrayFlat,axis=0))


@jax.jit
def ListMatrixize(FlatArray):

	""" Matrixize FlatArray & Organize It Listwise """

	global Rows,Cum

	L=len(Rows)
	return [jnp.asarray(jnp.reshape(jnp.take(FlatArray,jnp.arange(Cum[l],Cum[l+1])),(Rows[l],(Cum[l+1]-Cum[l])//Rows[l]))) for l in range(L)]
