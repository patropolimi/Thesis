#! /usr/bin/python3

from Adaptive.PINN_Problems import *


P=TemplateParameter('P')


class Resolutor_Adaptive(P,metaclass=Template[P]):

	""" Resolutor Of Differential Problem P Upon Adaptive PINN

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
			Grad_Pre=np.copy(Grad_Post)
			Iteration+=1
			print('L-BFGS: (%d/%d)' %(Iteration,LBFGS_MaxSteps),end='\r')


	def Iterations_Learning(self):

		""" Adaptive Learning Iterations """

		return (100*(2**(self.Architecture['Hidden_Layers']))*(np.sqrt(np.sqrt(np.sum(self.Architecture['Neurons_Per_Layer']))+self.Domain['Number_Residuals'])))


	def Main(self,Parameters=None):

		""" Main Function Handling Adaptive Learning """

		Old_Self=copy.deepcopy(self)
		Current_PoolValues_And_BC={'PoolValues': (Old_Self.Data['Source'](Old_Self.Pool_Residuals)-Old_Self.Equation(Old_Self.Pool_Residuals,Old_Self.Architecture['W']))**2,'BC': Old_Self.BC(Old_Self.BC_Default_X(),Old_Self.Architecture['W'])}
		[Continue,Salient_Cost_History]=[True,[(jnp.mean(Current_PoolValues_And_BC['PoolValues'])+Current_PoolValues_And_BC['BC']).item()]]
		while (Continue):
			Iterations=self.Iterations_Learning()
			[ADAM_Steps,LBFGS_MaxSteps]=[int(Iterations/10),int(9*Iterations/10)]
			self.Learn(ADAM_Steps,LBFGS_MaxSteps,Parameters)
			Current_PoolValues_And_BC={'PoolValues': (self.Data['Source'](self.Pool_Residuals)-self.Equation(self.Pool_Residuals,self.Architecture['W']))**2,'BC': self.BC(self.BC_Default_X(),self.Architecture['W'])}
			Salient_Cost_History+=[(jnp.mean(Current_PoolValues_And_BC['PoolValues'])+Current_PoolValues_And_BC['BC']).item()]
			Target=jnp.mean(Current_PoolValues_And_BC['PoolValues'])/self.PDE(self.PDE_Default_X(),self.Architecture['W'])
			if ((Salient_Cost_History[-2])>(Salient_Cost_History[-1]) or (self.Adaptivity['Force_First_Iterations'])>0):
				self.Adaptivity['Force_First_Iterations']=max(0,self.Adaptivity['Force_First_Iterations']-1)
				Old_Self=copy.deepcopy(self)
				if (((Salient_Cost_History[-2])/(Salient_Cost_History[-1]))>(self.Adaptivity['NoAction_Threshold'])):
					print('\nNo-Action Threshold Activated: Performing More Iterations Without Need Of Adaptive Features')
				else:
					print('\nInactive No-Action Threshold')
					if ((Target)<(self.Adaptivity['GRW_Threshold'])):
						self.Growing()
						self.Adaptivity['GRW_Threshold']=np.sqrt(self.Adaptivity['GRW_Threshold'])
					else:
						print('Skip Growing')
					if ((Target)>(self.Adaptivity['RAR_Threshold'])):
						self.Residual_Adaptive_Refinement(Current_PoolValues_And_BC['PoolValues'])
						self.Adaptivity['RAR_Threshold']=self.Adaptivity['RAR_Threshold']**2
					else:
						print('Skip Residual Adaptive Refinement')
			else:
				self=copy.deepcopy(Old_Self)
				Salient_Cost_History.pop()
				Continue=False
				print('\nPerformance Worsened: Ending')
		return Salient_Cost_History
