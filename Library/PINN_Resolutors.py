#! /usr/bin/python3

from PINN_Problems import *


P=TemplateParameter('P')


class Resolutor_ADAM_LBFGSB(P,metaclass=Template[P]):

	""" Resolutor Of Differential Problem P

		Optimizers:
		- ADAM
		- L-BFGS-B """


	def ADAM(self,Epochs,Batch,W=None,XR=None,XB=None,a=1e-3,b1=0.9,b2=0.999,d=1e-8):

		""" ADAM Optimization Method """

		if W is None:
			W=self.Weights
		if XR is None:
			XR=self.Residual_Points
		if XB is None:
			XB=self.Boundary_Lists

		Random_Seed()
		L=len(W)
		NR=XR.shape[1]
		m=[np.zeros_like(w) for w in W]
		v=[np.zeros_like(w) for w in W]
		for i in range(Epochs):
			IdxsR=np.random.choice(NR,Batch)
			g=self.Gradient_Cost(W,XR[:,IdxsR],XB)
			for l in range(L):
				m[l]=(b1*m[l]+(1-b1)*g[l])/(1-b1**(i+1))
				v[l]=(b2*v[l]+(1-b2)*g[l]*g[l])/(1-b2**(i+1))
				W[l]-=(a*m[l]/(np.sqrt(v[l])+d))


	def LBFGSB(self,MaxEpochs,W=None,XR=None,XB=None):

		""" L-BFGS-B Optimization Method """

		if W is None:
			W=self.Weights
		if XR is None:
			XR=self.Residual_Points
		if XB is None:
			XB=self.Boundary_Lists

		W,W_Rows,W_Cum=Flatten(W)
		W,_,_=lbfgsb(self.Cost,W,fprime=self.Gradient_Cost,args=(W_Rows,W_Cum,XR,XB),maxiter=MaxEpochs)
		W=ListMatrixize(W,W_Rows,W_Cum)
		return W


	def Learn(self,Iters_ADAM,Batch_ADAM,MaxIters_LBFGSB,W=None,XR=None,XB=None):

		""" Learn Prompter """

		FinalStep=(W==None)

		ADAM(Iters_ADAM,Batch_ADAM,W,XR,XB)
		W,_,_=LBFGSB(MaxIters_LBFGSB,W,XR,XB)
		if (FinalStep):
			self.Weights=W
		return W
