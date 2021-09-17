#! /usr/bin/python3

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

def MSE(W,X,Y):
	DNN_Out=DNN(W,X)
	return jnp.mean(jnp.sum((DNN_Out-Y)**2,axis=0))

def Tanh(X):
	return jnp.tanh(X)

def ReLU(X):
	return jnp.maximum(0,X)

def Sigmoid(X):
	return 1.0/(1.0+jnp.exp(X))

def f1(x):
	return np.sin(0.5*np.pi*x)

def f2(x):
	return np.sin(1.5*np.pi*x)

def f3(x):
	return 0.3*np.sin(10*np.pi*x)+0.5*np.cos(np.pi*x)+0.3*np.sin(np.pi*0.1*(x-0.5))

def Glorot(Layers_Size,Seed):
	np.random.seed(Seed)
	Weights=list()
	L=len(Layers_Size)-1
	for i in range(L):
		Weights.append(np.concatenate((np.random.randn(Layers_Size[i+1],Layers_Size[i])*np.sqrt(2/(Layers_Size[i+1]+Layers_Size[i])),
	                               np.zeros((Layers_Size[i+1],1))),axis=1))
		if (i!=(L-1)):
			Weights.append(np.array([1.0/(i+1)]))
	return Weights

def DNN(Weights,X):
	L=(len(Weights)+1)//2
	Y=X
	for l in range(L):
		Y=Weights[2*l][:,:-1]@Y+Weights[2*l][:,-1:]
		if (l!=(L-1)):
			Y=Sigma((l+1)*Weights[2*l+1]*Y)
	return Y
DNN=jax.jit(DNN)

def RMSPROP(X,Y,W,n_e,nu,bs,d,ro):
	L=len(W)
	N=(X.shape)[1]
	v=[np.zeros(w.shape) for w in W]
	for i in range(n_e):
		Idxs=np.random.choice(N,bs)
		g=Grad_J(W,X[:,Idxs],Y[:,Idxs])
		for l in range(L):
			v[l]=ro*v[l]+(1.0-ro)*(g[l]*g[l])
			W[l]-=(nu/(d+np.sqrt(v[l])))*g[l]