#! /usr/bin/python3

from PINN_Problems import *
from type_templating import Template,TemplateParameter


P=TemplateParameter('P')


class Resolutor_ADAM_LBFGSB(P,metaclass=Template[P]):

	""" Resolutor Of Differential Problem P

		Optimizers:
		- ADAM
		- L-BFGS-B """


	def ADAM(self,Epochs,Batch,a=1e-3,b1=0.9,b2=0.999,d=1e-8):
		Uts.Random_Seed()
		L=len(self.Weights)
		NR=self.Number_Residuals
		m=[np.zeros_like(w) for w in self.Weights]
		v=[np.zeros_like(w) for w in self.Weights]
		for i in range(Epochs):
			IdxsR=np.random.choice(NR,Batch)
			g=self.Gradient_Cost(self.Weights,self.Residual_Points[:,IdxsR],self.Boundary_Lists)
			for l in range(L):
				m[l]=(b1*m[l]+(1-b1)*g[l])/(1-b1**(i+1))
				v[l]=(b2*v[l]+(1-b2)*g[l]*g[l])/(1-b2**(i+1))
				self.Weights[l]-=(a*m[l]/(np.sqrt(v[l])+d))


	def L_BFGS_B():


	def Learn(self,Iters_ADAM,Iters_LBFGSB,Batch_ADAM,Batch_LBFGSB):
		print('ADAM Optimizer Working...')
		self.ADAM(Iters_ADAM,Batch_ADAM)
		print('L-BFGS-B Optimizer Working...')
		self.L_BFGS_B(Iters_LBFGSB,Batch_LBFGSB)
