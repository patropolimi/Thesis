#! /usr/bin/python3

from PINN_Problems import *


P=TemplateParameter('P')


class Resolutor_ADAM_BFGS(P,metaclass=Template[P]):

	""" Resolutor Of Differential Problem P

		Optimizers:
		- ADAM -> First Order
		- BFGS -> Super-Linear Order """


	ADAM_Default={'Alpha': 1e-3,'Beta': 0.9,'Gamma': 0.999,'Delta': 1e-7}
	BFGS_Default={'GradTol': 1e-5,'AlphaZero': 10.0,'C': 0.5,'Tau': 0.5}


	def ADAM(self,Epochs,Batch,Parameters=None):

		""" ADAM Optimization Method """

		if Parameters is None:
			[a,b,c,d]=[self.ADAM_Default['Alpha'],self.ADAM_Default['Beta'],self.ADAM_Default['Gamma'],self.ADAM_Default['Delta']]
		else:
			[a,b,c,d]=[Parameters['Alpha'],Parameters['Beta'],Parameters['Gamma'],Parameters['Delta']]

		Hist=[]
		np.random.seed(Random_Seed())
		L=len(self.Weights)
		m=[np.zeros_like(w) for w in self.Weights]
		v=[np.zeros_like(w) for w in self.Weights]
		XR={'Residual_Points': self.Residual_Points,'Number_Residuals': Batch}
		for i in range(Epochs):
			if (i%10==0):
				Hist+=[self.Cost(self.Weights,self.PDE_Default_X(),self.BC_Default_X())]
			IdxsR=np.random.choice(self.Number_Residuals,Batch)
			XR['Residual_Points']=self.Residual_Points[:,IdxsR]
			g=self.Gradient_Cost(self.Weights,XR,self.BC_Default_X())
			for l in range(L):
				m[l]=(b*m[l]+(1-b)*g[l])
				v[l]=(c*v[l]+(1-c)*g[l]*g[l])
				self.Weights[l]-=(a*(m[l]/(1-b**(i+1))))/(np.sqrt(v[l]/(1-c**(i+1)))+d)
		return Hist


	def BFGS(self,MaxEpochs,Parameters=None):

		""" BFGS Optimization Method """

		if Parameters is None:
			[GradTol,AlphaZero,C,Tau]=[self.BFGS_Default['GradTol'],self.BFGS_Default['AlphaZero'],self.BFGS_Default['C'],self.BFGS_Default['Tau']]
		else:
			[GradTol,AlphaZero,C,Tau]=[Parameters['GradTol'],Parameters['AlphaZero'],Parameters['C'],Parameters['Tau']]

		@jax.jit
		def Cost_BFGS(W,Alpha,D):

			""" Helper Cost For BFGS """

			WM=ListMatrixize(W+Alpha*D)
			return self.PDE(self.PDE_Default_X(),WM)+self.BC(self.BC_Default_X(),WM)

		Grad_Cost_BFGS=jax.jit(jax.grad(Cost_BFGS))

		def Line_Search(W,Alpha_Pre,D,M,Cost_Pre):

			""" Two-Way Backtracking Line Search Algorithm """

			A=Alpha_Pre
			T=-C*M
			Cost_Post=Cost_BFGS(W,A,D)
			if ((Cost_Pre-Cost_Post)>=(A*T)):
				while ((Cost_Pre-Cost_Post)>=(A*T) and (A<=AlphaZero)):
					A/=Tau
					Cost_Post=Cost_BFGS(W,A,D)
			else:
				while ((Cost_Pre-Cost_Post)<(A*T)):
					A*=Tau
					Cost_Post=Cost_BFGS(W,A,D)
			return A

		Hist=[]
		W=np.asarray(Flatten(self.Weights))
		Z=np.zeros_like(W)
		N=W.shape[0]
		I=np.eye(N)
		B=np.copy(I)
		Iters=0
		Grad_Pre=Grad_Cost_BFGS(W,0.0,Z)
		Alpha=AlphaZero
		while (not(np.linalg.norm(Grad_Pre)<GradTol) and (Iters<MaxEpochs)):
			Cost=Cost_BFGS(W,0.0,Z)
			if (Iters%10==0):
				Hist+=[Cost]
			Direction=-(B)@(Grad_Pre)
			Alpha=Line_Search(W,Alpha,Direction,np.inner(Grad_Pre,Direction),Cost)
			S=Alpha*Direction
			W+=S
			Grad_Post=Grad_Cost_BFGS(W,0.0,Z)
			Y=Grad_Post-Grad_Pre
			Denominator=np.inner(S,Y)
			B=(I-(S[:,None])@(Y[None,:])/Denominator)@(B)@(I-(Y[:,None])@(S[None,:])/Denominator)+((S[:,None])@(S[None,:])/Denominator)
			Grad_Pre=np.copy(Grad_Post)
			Iters+=1
		self.Weights=ListMatrixize(W)
		return Hist


	def Learn(self,ADAM_Steps,ADAM_Batch,BFGS_MaxSteps,ADAM_Params=None,BFGS_Params=None):

		""" Learning Execution """

		Tot_Hist=[]
		if (ADAM_Steps):
			print("ADAM Progressing ... ")
			Tot_Hist+=self.ADAM(ADAM_Steps,ADAM_Batch,ADAM_Params)
		if (BFGS_MaxSteps):
			print("BFGS Progressing ... ")
			Tot_Hist+=self.BFGS(BFGS_MaxSteps,BFGS_Params)
		return Tot_Hist
