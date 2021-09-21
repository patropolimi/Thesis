#! /usr/bin/python3

from PINN_Problems import *


P=TemplateParameter('P')


class Resolutor_ADAM_BFGS(P,metaclass=Template[P]):

	""" Resolutor Of Differential Problem P

		Optimizers:
		- ADAM -> First Order
		- BFGS -> Super-Linear Order """


	ADAM_Default={'Alpha': 1e-3,'Beta': 0.9,'Gamma': 0.999,'Delta': 1e-8}
	BFGS_Default={'GradTol': 1e-5,'LSNu': 1e-3,'LSMax': 50,'LSTol': 1e-5}


	def ADAM(self,Epochs,Batch,Parameters=None):

		""" ADAM Optimization Method """

		if Parameters is None:
			[a,b,c,d]=[self.ADAM_Default['Alpha'],self.ADAM_Default['Beta'],self.ADAM_Default['Gamma'],self.ADAM_Default['Delta']]
		else:
			[a,b,c,d]=[Parameters['Alpha'],Parameters['Beta'],Parameters['Gamma'],Parameters['Delta']]

		np.random.seed(Random_Seed())
		L=len(self.Weights)
		m=[np.zeros_like(w) for w in self.Weights]
		v=[np.zeros_like(w) for w in self.Weights]
		XR={'Residual_Points': self.Residual_Points,'Number_Residuals': Batch}
		for i in range(Epochs):
			IdxsR=np.random.choice(self.Number_Residuals,Batch)
			XR['Residual_Points']=self.Residual_Points[:,IdxsR]
			g=self.Gradient_Cost(self.Weights,XR,self.BC_Default_X())
			for l in range(L):
				m[l]=(b*m[l]+(1-b)*g[l])/(1-b**(i+1))
				v[l]=(c*v[l]+(1-c)*g[l]*g[l])/(1-c**(i+1))
				self.Weights[l]-=(a*m[l]/(np.sqrt(v[l])+d))
		print("Current Cost: %.3e " %(self.Cost(self.Weights,self.PDE_Default_X(),self.BC_Default_X())))


	def BFGS(self,MaxEpochs,Parameters=None):

		""" BFGS Optimization Method """

		if Parameters is None:
			[GradTol,LSNu,LSMax,LSTol]=[self.BFGS_Default['GradTol'],self.BFGS_Default['LSNu'],self.BFGS_Default['LSMax'],self.BFGS_Default['LSTol']]
		else:
			[GradTol,LSNu,LSMax,LSTol]=[Parameters['GradTol'],Parameters['LSNu'],Parameters['LSMax'],Parameters['LSTol']]

		@jax.jit
		def Cost_BFGS(W,Alpha=0,D=0):

			""" Helper Cost For BFGS """

			WM=ListMatrixize(W+Alpha*D)
			return self.PDE(self.PDE_Default_X(),WM)+self.BC(self.BC_Default_X(),WM)

		Grad_Cost_BFGS_W=jax.jit(jax.grad(Cost_BFGS))
		Grad_Cost_BFGS_A=jax.jit(jax.grad(Cost_BFGS,argnums=1))

		def Line_Search(W,Alpha,D):

			""" Minimum Along Direction D

				Optimizer -> Gradient Descent """

			A_Pre=Alpha
			Its=0
			Condition=True
			while (Condition and (Its<LSMax)):
				G=Grad_Cost_BFGS_A(W,A_Pre,D)
				A_Post=A_Pre-LSNu*G
				Condition=((abs(G)>LSTol) and ((abs(A_Post-A_Pre)/abs(A_Pre))>LSTol))
				A_Pre=A_Post
				Its+=1
			return A_Post

		W=np.asarray(Flatten(self.Weights))
		N=W.shape[0]
		I=np.eye(N)
		B=np.copy(I)
		Iters=0
		Grad_Pre=Grad_Cost_BFGS_W(W)
		while (not(np.linalg.norm(Grad_Pre)<GradTol) and (Iters<MaxEpochs)):
			Direction=-(B)@(Grad_Pre)
			Alpha=Line_Search(W,LSNu,Direction)
			S=Alpha*Direction
			W+=S
			Grad_Post=Grad_Cost_BFGS_W(W)
			Y=Grad_Post-Grad_Pre
			Denominator=np.inner(S,Y)
			B=(I-(S[:,None])@(Y[None,:])/Denominator)@(B)@(I-(Y[:,None])@(S[None,:])/Denominator)+((S[:,None])@(S[None,:])/Denominator)
			Grad_Pre=np.copy(Grad_Post)
			Iters+=1
		self.Weights=ListMatrixize(W)
		print("Current Cost: %.3e " %(self.Cost(self.Weights,self.PDE_Default_X(),self.BC_Default_X())))


	def Learn(self,ADAM_Calls,ADAM_Steps,ADAM_Batch,BFGS_MaxSteps,ADAM_Params=None,BFGS_Params=None):

		""" Learning Prompter """

		for i in range(ADAM_Calls):
			print("ADAM Progressing ... ")
			self.ADAM(ADAM_Steps[i],ADAM_Batch,ADAM_Params)
		print("BFGS Progressing ... ")
		self.BFGS(BFGS_MaxSteps,BFGS_Params)
