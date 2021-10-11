#! /usr/bin/python3

from Basic.PINN_Problems import *


P=TemplateParameter('P')


class Resolutor_Basic(P,metaclass=Template[P]):

	""" Resolutor Of Differential Problem P Upon Basic PINN

		Optimizers:
		- ADAM -> First Order
		- L-BFGS -> Super-Linear Order """


	ADAM_Default={'Alpha': 1e-3,'Beta': 0.9,'Gamma': 0.999,'Delta': 1e-8}
	LBFGS_Default={'Memory': 10,'GradTol': 1e-6,'AlphaTol': 1e-6,'StillTol': 10,'AlphaZero': 10.0,'C': 0.5,'T': 0.5,'Eps': 1e-8}


	def ADAM(self,Epochs,Batch_Fraction,Parameters=None):

		""" ADAM Optimization Method """

		if Parameters is None:
			[a,b,c,d]=[self.ADAM_Default['Alpha'],self.ADAM_Default['Beta'],self.ADAM_Default['Gamma'],self.ADAM_Default['Delta']]
		else:
			[a,b,c,d]=[Parameters['Alpha'],Parameters['Beta'],Parameters['Gamma'],Parameters['Delta']]

		np.random.seed(Random_Seed())
		Cost_Hist=[self.Cost(self.Architecture['W'],self.PDE_Default_X(),self.BC_Default_X())]
		m=np.zeros_like(self.Architecture['W'])
		v=np.zeros_like(m)
		print('ADAM: (0/%d)' %(Epochs),end='\r')
		for i in range(Epochs):
			Res_Batch=int(np.ceil(Batch_Fraction*self.Domain['Number_Residuals']))
			IdxsR=np.random.choice(self.Domain['Number_Residuals'],Res_Batch)
			g=self.Gradient_Cost(self.Architecture['W'],self.Domain['Residual_Points'][:,IdxsR],self.BC_Default_X())
			m=b*m+(1-b)*g
			v=c*v+(1-c)*(g**2)
			self.Architecture['W']-=(a*(m/(1-b**(i+1))))/(np.sqrt(v/(1-c**(i+1)))+d)
			print('ADAM: (%d/%d)' %(i+1,Epochs),end='\r')
			Cost_Hist+=[self.Cost(self.Architecture['W'],self.PDE_Default_X(),self.BC_Default_X())]
		return Cost_Hist


	def Line_Search(self,W,D,M,C,T,Cost_Pre,Alpha_Pre,AlphaZero,Eps):

		""" Two-Way Backtracking Line Search Algorithm For LBFGS """

		Alpha=Alpha_Pre
		Z=-C*M
		Cost_Post=self.Cost(W+Alpha*D,self.PDE_Default_X(),self.BC_Default_X())
		if ((Cost_Pre-Cost_Post)>=np.abs(Alpha*Z)):
			while ((Cost_Pre-Cost_Post)>=np.abs(Alpha*Z) and (Alpha<=AlphaZero)):
				Alpha/=T
				Cost_Post=self.Cost(W+Alpha*D,self.PDE_Default_X(),self.BC_Default_X())
			Alpha*=T
		else:
			while ((Cost_Pre-Cost_Post)<(np.abs(Alpha*Z)-Eps)):
				Alpha*=T
				Cost_Post=self.Cost(W+Alpha*D,self.PDE_Default_X(),self.BC_Default_X())
		return Alpha


	@partial(jax.jit,static_argnums=(0))
	def Two_Loops_Recursion(self,Grad,Mem_DeltaW,Mem_DeltaGrad,Mem_Ro):

		""" Two-Loop Recursion Algorithm For LBFGS """

		M=len(Mem_DeltaW)
		C=jnp.inner(Mem_DeltaW[0],Mem_DeltaGrad[0])/jnp.inner(Mem_DeltaGrad[0],Mem_DeltaGrad[0])
		A=M*[0]
		D=-Grad
		for i in range(M):
			A[i]=jnp.inner(Mem_DeltaW[i],D)/Mem_Ro[i]
			D-=A[i]*Mem_DeltaGrad[i]
		D*=C
		for i in reversed(range(M)):
			B=jnp.inner(Mem_DeltaGrad[i],D)/Mem_Ro[i]
			D+=Mem_DeltaW[i]*(A[i]-B)
		return D


	@partial(jax.jit,static_argnums=(0))
	def Update_B(self,B,S,Y,I,Denominator):

		""" Helper To Update Inverse Hessian Approximation For BFGS """

		M1=-(S[:,None])@(Y[None,:])/Denominator
		M2=(S[:,None])@(S[None,:])/Denominator
		New_B=I+M1
		New_B=(New_B)@(B)
		New_B=(New_B)@(I+M1.T)
		New_B+=M2
		return New_B


	def LBFGS(self,MaxEpochs,Parameters=None):

		""" L-BFGS Optimization Method

			Procedure:
			- First Iterations To Initialize Memory Vectors -> BFGS
			- Remaining Iterations -> L-BFGS """

		LBFGS_Default={'Memory': 10,'GradTol': 1e-6,'AlphaTol': 1e-6,'StillTol': 10,'AlphaZero': 10.0,'C': 0.5,'T': 0.5,'Eps': 1e-8}

		if Parameters is None:
			[Memory,GradTol,AlphaTol,StillTol,AlphaZero,C,T,Eps]=[self.LBFGS_Default['Memory'],self.LBFGS_Default['GradTol'],self.LBFGS_Default['AlphaTol'],self.LBFGS_Default['StillTol'],self.LBFGS_Default['AlphaZero'],self.LBFGS_Default['C'],self.LBFGS_Default['T'],self.LBFGS_Default['Eps']]
		else:
			[Memory,GradTol,AlphaTol,StillTol,AlphaZero,C,T,Eps]=[Parameters['Memory'],Parameters['GradTol'],Parameters['AlphaTol'],Parameters['StillTol'],Parameters['AlphaZero'],Parameters['C'],Parameters['T'],Parameters['Eps']]

		N=self.Architecture['W'].shape[0]
		Memory_DeltaW=Cyclic_Deque(Memory,np.zeros(N))
		Memory_DeltaGrad=Cyclic_Deque(Memory,np.zeros(N))
		Memory_Ro=Cyclic_Deque(Memory,0)
		I=np.eye(N)
		B=np.copy(I)
		Cost_Pre=self.Cost(self.Architecture['W'],self.PDE_Default_X(),self.BC_Default_X())
		Grad_Pre=self.Gradient_Cost(self.Architecture['W'],self.PDE_Default_X(),self.BC_Default_X())
		Cost_Hist=[Cost_Pre]
		Consecutive_Still=0
		Alpha=AlphaZero
		Iteration=0
		Methods=['BFGS','L-BFGS']
		for Method in Methods:
			print('\n'+Method+': (%d/%d)' %(Iteration,max(Memory,MaxEpochs*(Method=='L-BFGS'))),end='\r')
			while (not(jnp.linalg.norm(Grad_Pre)<GradTol) and not(Cost_Pre<EpsMachine) and (Iteration<max(Memory,(Method=='L-BFGS')*MaxEpochs)) and (Consecutive_Still<StillTol)):
				if (Method=='BFGS'):
					Direction=-(B)@(Grad_Pre)
				elif (Method=='L-BFGS'):
					Direction=self.Two_Loops_Recursion(Grad_Pre,Memory_DeltaW.List(),Memory_DeltaGrad.List(),Memory_Ro.List())
				if (not(jnp.all(jnp.isfinite(Direction)))):
					print('Invalid Search Direction Encountered')
					break
				Alpha=self.Line_Search(self.Architecture['W'],Direction,jnp.inner(Grad_Pre,Direction),C,T,Cost_Pre,Alpha,AlphaZero,Eps)
				S=Alpha*Direction
				self.Architecture['W']+=S
				Cost_Post=self.Cost(self.Architecture['W'],self.PDE_Default_X(),self.BC_Default_X())
				Grad_Post=self.Gradient_Cost(self.Architecture['W'],self.PDE_Default_X(),self.BC_Default_X())
				Y=Grad_Post-Grad_Pre
				Denominator=jnp.inner(S,Y)
				Memory_DeltaW.Insert(S)
				Memory_DeltaGrad.Insert(Y)
				Memory_Ro.Insert(Denominator)
				Cost_Hist+=[Cost_Post]
				Cost_Pre=Cost_Post
				Grad_Pre=np.copy(Grad_Post)
				Iteration+=1
				if (np.abs(Alpha)<AlphaTol):
					Consecutive_Still+=1
				else:
					Consecutive_Still=0
				if (Method=='BFGS'):
					if (Iteration<Memory):
						B=self.Update_B(B,S,Y,I,Denominator)
					else:
						del I,B
				print(Method+': (%d/%d)' %(Iteration,max(Memory,MaxEpochs*(Method=='L-BFGS'))),end='\r')
		return Cost_Hist


	def Learn(self,ADAM_Steps,ADAM_BatchFraction,LBFGS_MaxSteps,ADAM_Params=None,LBFGS_Params=None):

		""" Learning Execution """

		return self.ADAM(ADAM_Steps,ADAM_BatchFraction,ADAM_Params)[:-1]+self.LBFGS(LBFGS_MaxSteps,LBFGS_Params)
