#! /usr/bin/python3

from PINN_Problems import *


P=TemplateParameter('P')


class Resolutor_ADAM_BFGS(P,metaclass=Template[P]):

	""" Resolutor Of Differential Problem P

		Optimizers:
		- ADAM -> First Order
		- BFGS -> Super-Linear Order """


	ADAM_Default={'Alpha': 1e-3,'Beta': 0.9,'Gamma': 0.999,'Delta': 1e-7}
	BFGS_Default={'GradTol': 1e-6,'AlphaZero': 10.0,'C': 0.5,'Tau': 0.5,'StillTol': 10,'AlphaTol': 1e-10,'Eps': 1e-12}


	def ADAM(self,Epochs,Batch,Parameters=None):

		""" ADAM Optimization Method """

		if Parameters is None:
			[a,b,c,d]=[self.ADAM_Default['Alpha'],self.ADAM_Default['Beta'],self.ADAM_Default['Gamma'],self.ADAM_Default['Delta']]
		else:
			[a,b,c,d]=[Parameters['Alpha'],Parameters['Beta'],Parameters['Gamma'],Parameters['Delta']]

		np.random.seed(Random_Seed())
		Hist=[self.Cost(self.Weights,self.PDE_Default_X(),self.BC_Default_X())]
		L=len(self.Weights)
		m=[np.zeros_like(w) for w in self.Weights]
		v=[np.zeros_like(w) for w in self.Weights]
		XR={'Residual_Points': self.Residual_Points,'Number_Residuals': Batch}
		for i in range(Epochs):
			IdxsR=np.random.choice(self.Number_Residuals,Batch)
			XR['Residual_Points']=self.Residual_Points[:,IdxsR]
			g=self.Gradient_Cost(self.Weights,XR,self.BC_Default_X())
			for l in range(L):
				m[l]=(b*m[l]+(1-b)*g[l])
				v[l]=(c*v[l]+(1-c)*g[l]*g[l])
				self.Weights[l]-=(a*(m[l]/(1-b**(i+1))))/(np.sqrt(v[l]/(1-c**(i+1)))+d)
			Hist+=[self.Cost(self.Weights,self.PDE_Default_X(),self.BC_Default_X())]
		return Hist


	def BFGS(self,MaxEpochs,Parameters=None):

		""" BFGS Optimization Method """

		if Parameters is None:
			[GradTol,AlphaZero,C,Tau,StillTol,AlphaTol,Eps]=[self.BFGS_Default['GradTol'],self.BFGS_Default['AlphaZero'],self.BFGS_Default['C'],self.BFGS_Default['Tau'],self.BFGS_Default['StillTol'],self.BFGS_Default['AlphaTol'],self.BFGS_Default['Eps']]
		else:
			[GradTol,AlphaZero,C,Tau,StillTol,AlphaTol,Eps]=[Parameters['GradTol'],Parameters['AlphaZero'],Parameters['C'],Parameters['Tau'],Parameters['StillTol'],Parameters['AlphaTol'],Parameters['Eps']]

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
			if ((Cost_Pre-Cost_Post)>=np.abs(A*T)):
				while ((Cost_Pre-Cost_Post)>=np.abs(A*T) and (A<=AlphaZero)):
					A/=Tau
					Cost_Post=Cost_BFGS(W,A,D)
				A*=Tau
			else:
				while ((Cost_Pre-Cost_Post)<(np.abs(A*T)-Eps)):
					A*=Tau
					Cost_Post=Cost_BFGS(W,A,D)
			return A

		@jax.jit
		def Update_B(B,S,Y,I,Denominator):

			""" Helper To Update B (Inverse Hessian Approximation) """

			M1=-(S[:,None])@(Y[None,:])/Denominator
			M2=(S[:,None])@(S[None,:])/Denominator
			New_B=I+M1
			New_B=(New_B)@(B)
			New_B=(New_B)@(I+M1.T)
			New_B+=M2
			return New_B

		Hist=[]
		W=np.asarray(Flatten(self.Weights))
		N=W.shape[0]
		Z=np.zeros_like(W)
		I=np.eye(N)
		B=np.copy(I)
		Cost_Pre=Cost_BFGS(W,0.0,Z)
		Grad_Pre=Grad_Cost_BFGS(W,0.0,Z)
		Alpha=AlphaZero
		Consecutive_Still=0
		Iters=0
		while (not(jnp.linalg.norm(Grad_Pre)<GradTol) and (Iters<MaxEpochs) and (Consecutive_Still<StillTol)):
			Direction=-(B)@(Grad_Pre)
			Alpha=Line_Search(W,Alpha,Direction,jnp.inner(Grad_Pre,Direction),Cost_Pre)
			S=Alpha*Direction
			W+=S
			Cost_Post=Cost_BFGS(W,0.0,Z)
			Grad_Post=Grad_Cost_BFGS(W,0.0,Z)
			Y=Grad_Post-Grad_Pre
			Denominator=jnp.inner(S,Y)
			B=Update_B(B,S,Y,I,Denominator)
			if (not(jnp.all(jnp.isfinite(B)))):
				break
			if (Alpha<AlphaTol):
				Consecutive_Still+=1
			else:
				Consecutive_Still=0
			Hist+=[Cost_Pre]
			Cost_Pre=Cost_Post
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
