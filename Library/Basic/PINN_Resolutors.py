#! /usr/bin/python3

from Basic.PINN_Problems import *


P=TemplateParameter('P')


class Resolutor_Basic(P,metaclass=Template[P]):

	""" Resolutor Of Differential Problem P Upon Basic PINN

		Optimizers:
		- ADAM
		- LBFGS """


	Default={'Alpha': 1e-3,'Beta': 0.9,'Gamma': 0.999,'Delta': 1e-8,'Memory': 10,'GradTol': 1e-6,'AlphaZero': 10.0,'C': 0.5,'T': 0.5,'Eps': 1e-8}


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


	def Learn(self,ADAM_Steps,LBFGS_MaxSteps,Parameters=None):

		""" Learning Execution """

		if Parameters is None:
			[a,b,c,d,Memory,GradTol,AlphaZero,C,T,Eps]=[self.Default['Alpha'],self.Default['Beta'],self.Default['Gamma'],self.Default['Delta'],self.Default['Memory'],self.Default['GradTol'],self.Default['AlphaZero'],self.Default['C'],self.Default['T'],self.Default['Eps']]
		else:
			[a,b,c,d,Memory,GradTol,AlphaZero,C,T,Eps]=[Parameters['Alpha'],Parameters['Beta'],Parameters['Gamma'],Parameters['Delta'],Parameters['Memory'],Parameters['GradTol'],Parameters['AlphaZero'],Parameters['C'],Parameters['T'],Parameters['Eps']]

		Cost,Grad_Pre=self.Value_And_Gradient_Cost(self.Architecture['W'],self.PDE_Default_X(),self.BC_Default_X())
		Cost_History=[Cost.item()]
		N=self.Architecture['W'].shape[0]
		m=np.zeros(N)
		v=np.zeros(N)
		Memory_DeltaW=Cyclic_Deque(Memory,np.zeros(N))
		Memory_DeltaGrad=Cyclic_Deque(Memory,np.zeros(N))
		Memory_Ro=Cyclic_Deque(Memory,0)
		print('ADAM: (0/%d)' %(ADAM_Steps),end='\r')
		for Iteration in range(ADAM_Steps):
			m=b*m+(1-b)*Grad_Pre
			v=c*v+(1-c)*(Grad_Pre**2)
			DeltaW=-(a*(m/(1-b**(Iteration+1))))/(np.sqrt(v/(1-c**(Iteration+1)))+d)
			self.Architecture['W']+=DeltaW
			Cost,Grad_Post=self.Value_And_Gradient_Cost(self.Architecture['W'],self.PDE_Default_X(),self.BC_Default_X())
			Cost_History+=[Cost.item()]
			if ((Memory+Iteration)>=ADAM_Steps):
				Memory_DeltaW.Insert(DeltaW)
				Memory_DeltaGrad.Insert(Grad_Post-Grad_Pre)
				Memory_Ro.Insert(jnp.inner(DeltaW,Grad_Post-Grad_Pre))
			Grad_Pre=np.copy(Grad_Post)
			print('ADAM: (%d/%d)' %(Iteration+1,ADAM_Steps),end='\r')
		Alpha=AlphaZero
		Iteration=0
		print('\nL-BFGS: (0/%d)' %(LBFGS_MaxSteps),end='\r')
		while (not(jnp.linalg.norm(Grad_Pre)<GradTol) and not(Cost<EpsMachine) and (Iteration<LBFGS_MaxSteps)):
			Direction=self.Two_Loops_Recursion(Grad_Pre,Memory_DeltaW.List(),Memory_DeltaGrad.List(),Memory_Ro.List())
			if (not(jnp.all(jnp.isfinite(Direction)))):
				print('\nInvalid Search Direction Encountered')
				break
			Alpha=self.Line_Search(self.Architecture['W'],Direction,jnp.inner(Grad_Pre,Direction),C,T,Cost,Alpha,AlphaZero,Eps)
			S=Alpha*Direction
			self.Architecture['W']+=S
			Cost,Grad_Post=self.Value_And_Gradient_Cost(self.Architecture['W'],self.PDE_Default_X(),self.BC_Default_X())
			Y=Grad_Post-Grad_Pre
			Denominator=jnp.inner(S,Y)
			Memory_DeltaW.Insert(S)
			Memory_DeltaGrad.Insert(Y)
			Memory_Ro.Insert(Denominator)
			Cost_History+=[Cost.item()]
			Grad_Pre=np.copy(Grad_Post)
			Iteration+=1
			print('L-BFGS: (%d/%d)' %(Iteration,LBFGS_MaxSteps),end='\r')
		return Cost_History
