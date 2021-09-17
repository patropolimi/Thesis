#! /usr/bin/python3

import numpy as np
from smt.sampling_methods import FullFactorial,LHS,Random

class PINN:

	"""Adaptive Physics-Informed Neural Network

		Techniques:
		- Residual Adaptive Refinement
		- Pruning -> Weights
		- Growing -> Layers
		- Soft Attention
		
		Features:
		- Weights Initialization -> Glorot
		- Optimizer -> ADAM
		- Cost Function -> MSE
		- Domain -> Hyper-Rectangle
		- Boundary Conditions -> Dirichlet/Neumann (Possibly Periodic)"""

	Generator=np.random.default_rng(seed=time.time_ns())

	def Random_Seed():
		return Generator.integers(0,1e10)

	def __init__(self,D,DB,IN,NGR,SFR,Sigma,PDE,BC,IR,Dir_PTS=0,Neu_PTS=0,Per_PTS=0):
		np.random.seed(Random_Seed())
		self.Input_Dimension=D
		self.Domain_Boundaries=DB
		self.Initial_Neurons=IN
		self.Neuron_Growth_Rate=NGR
		self.Scaling_Factor_Rate=SFR
		self.Activation=Sigma
		self.PDE=PDE
		self.BC=BC	# Unire PDE & BC in un unico data structure da cui li posso chiamare (Equation.PDE Equation.BC?)
		self.Residual_Points=RAR(IR,0) # IR -> Massimo numero di Punti che stanno sopra la media (0, secondo parametro) che vanno presi
		self.Dirichlet_Points=Dir_PTS
		self.Neumann_Points=Neu_PTS
		self.Periodic_Points=Per_PTS
		self.W=[np.concatenate((np.random.randn(Initial_Neurons,D)*np.sqrt(2/(Initial_Neurons+D)),np.zeros((Initial_Neurons,1))),axis=1),
				np.concatenate((np.random.randn(1,Initial_Neurons)*np.sqrt(2/(Initial_Neurons+1)),np.zeros((1,1))),axis=1)]
		self.M=[np.ones_like(self.W[0]),np.ones_like(self.W[1])]
		self.N=[1.0]
		self.A=[1.0]
		self.Hidden_Layers=1

	def RAR(Num,Tol):
		Sample_Points=Random(self.Domain_Boundaries,Random_Seed())
		Add_Points=Sample_Points[self.PDE(Sample_Points)**2>Tol]
		self.Residual_Points
		self.Lambda_R

	def Prune(self):


	def Grow(self):
		Random_Seed()
		New_Neurons=np.ceil(Neuron_Growth_Rate**(self.Hidden_Layers)*Initial_Neurons)
		Last_Neurons=self.W[-1].shape[1]-1
		self.W[-1]=np.concatenate((np.random.randn(New_Neurons,Last_Neurons)*np.sqrt(2/(New_Neurons+Last_Neurons)),np.zeros((New_Neurons,1))),axis=1)
		self.M[-1]=np.ones_like(W[-1])
		self.W.append(np.concatenate((np.random.randn(1,New_Neurons)*np.sqrt(2/(New_Neurons+1)),np.zeros((1,1))),axis=1))
		self.M.append(np.ones_like(W[-1]))
		self.N.append(self.Scaling_Factor_Rate**(self.Hidden_Layers))
		self.A.append(1.0/self.Scaling_Factor_Rate**(self.Hidden_Layers))
		Hidden_Layers+=1

	def ADAM(self,X,n_e,bs,a=1e-3,b1=0.9,b2=0.999,d=1e-8):
		Random_Seed()
		L=len(self.W)
		S=X.shape[1]
		m_W=[np.zeros_like(w) for w in self.W]
		v_W=[np.zeros_like(w) for w in self.W]
		m_A=[0]*len(self.A)
		v_A=[0]*len(self.A)
		m_LR=np.zeros_like(Lambda_R)
		v_LR=np.zeros_like(Lambda_R)
		m_LB=np.zeros_like(Lambda_B)
		v_LB=np.zeros_like(Lambda_B)
		for i in range(n_e):
			Idxs=np.random.choice(S,bs)
			g_W=self.Grad_W(X,[:,Idxs])
			g_A=self.Grad_A(X,[:,Idxs])
			g_LR=self.Grad_LR(X,[:,Idxs])
			g_LB=self.Grad_LB(X,[:,Idxs])
			for l in range(L-1):
				m_W[l]=(b1*m_W[l]+(1-b1)*g_W[l])/(1-b1**(i+1))
				m_A[l]=(b1*m_A[l]+(1-b1)*g_A[l])/(1-b1**(i+1))
				v_W[l]=(b2*v_W[l]+(1-b2)*g_W[l]*g_W[l])/(1-b2**(i+1))
				v_A[l]=(b2*v_A[l]+(1-b2)*g_A[l]*g_A[l])/(1-b2**(i+1))
				self.W[l]-=(a*m_W[l]/(np.sqrt(v_W[l])+d))
				self.A[l]-=(a*m_A[l]/(np.sqrt(v_A[l])+d))
			m_W[L-1]=(b1*m_W[L-1]+(1-b1)*g_W[L-1])/(1-b1**(i+1))
			v_W[L-1]=(b2*v_W[L-1]+(1-b2)*g_W[L-1]*g_W[L-1])/(1-b2**(i+1))
			self.W[L-1]-=(a*m_W[L-1]/(np.sqrt(v_W[L-1])+d))
			m_LR=(b1*m_LR+(1-b1)*g_LR)/(1-b1**(i+1))
			m_LB=(b1*m_LB+(1-b1)*g_LB)/(1-b1**(i+1))
			v_LR=(b2*v_LR+(1-b2)*g_LR*g_LR)/(1-b2**(i+1))
			v_LB=(b2*v_LB+(1-b2)*g_LB*g_LB)/(1-b2**(i+1))
			self.Lambda_R+=(a*m_LR/(np.sqrt(v_LR)+d))
			self.Lambda_B+=(a*m_LB/(np.sqrt(v_LB)+d))

	def Cost():
		MSE_R=(1.0/NR)*jnp.norm(self.PDE(self.Residual_Points)*Lambda_R,2)**2
		MSE_B=(1.0/NB)*jnp.norm(self.BC(self.Dirichlet_Points,self.Neumann_Points,self.Periodic_Points)*Lambda_B,2)**2
		return MSE_R+MSE_B

	def Network(self,X,W,A,M,N):
		L=len(W)
		Y=X
		for l in range(L-1):
			Y=self.Activation(N[l]*A[l]*(((W[l][:,:-1]*M[l][:,:-1])@Y)+(W[l][:,-1:]*M[l][:,-1,:])))
		return ((W[L-1][:,:-1]*M[L-1][:,:-1])@Y)+(W[L-1][:,-1:]*M[L-1][:,-1,:])
	Network=jax.jit(Network)

	Grad_X=jax.grad(Cost,)
	Grad_W=jax.grad(Cost,)
	Grad_A=jax.grad(Cost,)
	Grad_X=jax.jit(Grad_X)
	Grad_W=jax.jit(Grad_W)
	Grad_A=jax.jit(Grad_A)

	def Learn:


