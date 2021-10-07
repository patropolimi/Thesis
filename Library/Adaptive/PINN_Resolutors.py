#! /usr/bin/python3

from Adaptive.PINN_Problems import *


P=TemplateParameter('P')


class Resolutor_ADAM_BFGS_Adaptive(P,metaclass=Template[P]):

	""" Resolutor Of Differential Problem P Upon Adaptive PINN

		Optimizers:
		- ADAM -> First Order
		- L-BFGS-Plus (BFGS + L-BFGS) -> Super-Linear Order """


	ADAM_Default={'Alpha': 1e-3,'Beta': 0.9,'Gamma': 0.999,'Delta': 1e-7}
	SuperLinear_Default={'GradTol': 1e-6,'AlphaZero': 10.0,'C': 0.5,'Tau': 0.5,'StillTol': 10,'AlphaTol': 1e-10,'Eps': 1e-12,'Memory': 10}


	def ADAM(self,Epochs,Batch,Parameters=None):

		""" ADAM Optimization Method """

		if Parameters is None:
			[a,b,c,d]=[self.ADAM_Default['Alpha'],self.ADAM_Default['Beta'],self.ADAM_Default['Gamma'],self.ADAM_Default['Delta']]
		else:
			[a,b,c,d]=[Parameters['Alpha'],Parameters['Beta'],Parameters['Gamma'],Parameters['Delta']]

		np.random.seed(Random_Seed())
		Cost_Hist=[self.Cost(self.Weights_On,self.A,self.N,self.L,self.PDE_Default_X(),self.BC_Default_X())]
		m=np.zeros_like(self.Weights_On)
		v=np.zeros_like(self.Weights_On)
		XR={'Residual_Points': self.Residual_Points,'Number_Residuals': Batch}
		W_Prev=np.copy(self.Weights_On)
		[Delta_W,SV]=2*[np.zeros_like(self.Weights_On)]
		for i in range(Epochs):
			if (i%self.RAR_Options['RAR_Step']==0):
				self.RAR()
			if (i%self.AAF_Options['AAF_Step']==0):
				self.Update_A()
			if (i%self.SAM_Options['SAM_Step']==0):
				self.Update_L()
			IdxsR=np.random.choice(self.Number_Residuals,Batch)
			XR['Residual_Points']=self.Residual_Points[:,IdxsR]
			g=self.Gradient_Cost_W(self.Weights_On,self.A,self.N,self.L,XR,self.BC_Default_X())
			m=b*m+(1-b)*g
			v=c*v+(1-c)*(g**2)
			self.Weights_On-=(a*(m/(1-b**(i+1))))/(np.sqrt(v/(1-c**(i+1)))+d)
			Delta_W=self.Weights_On-W_Prev
			SV+=Delta_W*g
			W_Prev=np.copy(self.Weights_On)
			Cost_Hist+=[self.Cost(self.Weights_On,self.A,self.N,self.L,self.PDE_Default_X(),self.BC_Default_X())]
		return {'Cost_History': Cost_Hist,'SV': SV}


	def Line_Search(self,W,Alpha_Pre,D,M,Cost_Pre,AlphaZero,C,Tau,Eps):

		""" Two-Way Backtracking Line Search Algorithm For BFGS & L-BFGS """

		Alpha=Alpha_Pre
		T=-C*M
		Cost_Post=self.Cost(W+Alpha*D,self.A,self.N,self.L,self.PDE_Default_X(),self.BC_Default_X())
		if ((Cost_Pre-Cost_Post)>=np.abs(Alpha*T)):
			while ((Cost_Pre-Cost_Post)>=np.abs(Alpha*T) and (Alpha<=AlphaZero)):
				Alpha/=Tau
				Cost_Post=self.Cost(W+Alpha*D,self.A,self.N,self.L,self.PDE_Default_X(),self.BC_Default_X())
			Alpha*=Tau
		else:
			while ((Cost_Pre-Cost_Post)<(np.abs(Alpha*T)-Eps)):
				Alpha*=Tau
				Cost_Post=self.Cost(W+Alpha*D,self.A,self.N,self.L,self.PDE_Default_X(),self.BC_Default_X())
		return Alpha


	@partial(jax.jit,static_argnums=(0))
	def Update_B(self,B,S,Y,I,Denominator):

		""" Helper To Update B (Inverse Hessian Approximation) For BFGS """

		M1=-(S[:,None])@(Y[None,:])/Denominator
		M2=(S[:,None])@(S[None,:])/Denominator
		New_B=I+M1
		New_B=(New_B)@(B)
		New_B=(New_B)@(I+M1.T)
		New_B+=M2
		return New_B


	def LBFGS_Plus(self,MaxEpochs,Parameters=None,Retrain=False):

		""" L-BFGS-Plus Optimization Method

			Procedure:
			- First Iterations To Initialize Memory Vectors -> BFGS
			- Remaining Iterations -> L-BFGS """

		if Parameters is None:
			[GradTol,AlphaZero,C,Tau,StillTol,AlphaTol,Eps,Memory]=[self.SuperLinear_Default['GradTol'],self.SuperLinear_Default['AlphaZero'],self.SuperLinear_Default['C'],self.SuperLinear_Default['Tau'],self.SuperLinear_Default['StillTol'],self.SuperLinear_Default['AlphaTol'],self.SuperLinear_Default['Eps'],self.SuperLinear_Default['Memory']]
		else:
			[GradTol,AlphaZero,C,Tau,StillTol,AlphaTol,Eps,Memory]=[Parameters['GradTol'],Parameters['AlphaZero'],Parameters['C'],Parameters['Tau'],Parameters['StillTol'],Parameters['AlphaTol'],Parameters['Eps'],Parameters['Memory']]

		N=self.Weights_On.shape[0]
		Memory_DeltaW=Cyclic_Deque(Memory)
		Memory_DeltaGrad=Cyclic_Deque(Memory)
		Memory_Ro=Cyclic_Deque(Memory)
		SV=np.zeros(N)
		I=np.eye(N)
		B=np.copy(I)
		Cost_Pre=self.Cost(self.Weights_On,self.A,self.N,self.L,self.PDE_Default_X(),self.BC_Default_X())
		Grad_Pre=self.Gradient_Cost_W(self.Weights_On,self.A,self.N,self.L,self.PDE_Default_X(),self.BC_Default_X())
		Cost_Hist=[Cost_Pre]
		Alpha=AlphaZero
		Consecutive_Still=0
		Iters=0
		while (not(jnp.linalg.norm(Grad_Pre)<GradTol) and not(Cost_Pre<EpsMachine) and (Iters<Memory) and (Consecutive_Still<StillTol)):
			if (Iters%self.RAR_Options['RAR_Step']==0 and not Retrain):
				self.RAR()
			if (Iters%self.AAF_Options['AAF_Step']==0 and not Retrain):
				self.Update_A()
			if (Iters%self.SAM_Options['SAM_Step']==0 and not Retrain):
				self.Update_L()
			Direction=-(B)@(Grad_Pre)
			Alpha=self.Line_Search(self.Weights_On,Alpha,Direction,jnp.inner(Grad_Pre,Direction),Cost_Pre,AlphaZero,C,Tau,Eps)
			S=Alpha*Direction
			self.Weights_On+=S
			Cost_Post=self.Cost(self.Weights_On,self.A,self.N,self.L,self.PDE_Default_X(),self.BC_Default_X())
			Grad_Post=self.Gradient_Cost_W(self.Weights_On,self.A,self.N,self.L,self.PDE_Default_X(),self.BC_Default_X())
			Y=Grad_Post-Grad_Pre
			Denominator=jnp.inner(S,Y)
			Memory_DeltaW.Insert(S)
			Memory_DeltaGrad.Insert(Y)
			Memory_Ro.Insert(Denominator)
			B=self.Update_B(B,S,Y,I,Denominator)
			Cost_Hist+=[Cost_Post]
			SV+=S*Grad_Pre
			Cost_Pre=Cost_Post
			Grad_Pre=np.copy(Grad_Post)
			Iters+=1
			if (np.abs(Alpha)<AlphaTol):
				Consecutive_Still+=1
			else:
				Consecutive_Still=0
			if (not(jnp.all(jnp.isfinite(B)))):
				break
		del I,B
		while (not(jnp.linalg.norm(Grad_Pre)<GradTol) and not(Cost_Pre<EpsMachine) and (Iters<MaxEpochs) and (Consecutive_Still<StillTol)):
			if (Iters%self.RAR_Options['RAR_Step']==0 and not Retrain):
				self.RAR()
			if (Iters%self.AAF_Options['AAF_Step']==0 and not Retrain):
				self.Update_A()
			if (Iters%self.SAM_Options['SAM_Step']==0 and not Retrain):
				self.Update_L()
			Direction=Two_Loops_Recursion(Grad_Pre,Memory_DeltaW,Memory_DeltaGrad,Memory_Ro)
			Alpha=self.Line_Search(W,Alpha,Direction,jnp.inner(Grad_Pre,Direction),Cost_Pre,AlphaZero,C,Tau,Eps)
			S=Alpha*Direction
			self.Weights_On+=S
			Cost_Post=self.Cost(self.Weights_On,self.A,self.N,self.L,self.PDE_Default_X(),self.BC_Default_X())
			Grad_Post=self.Gradient_Cost_W(self.Weights_On,self.A,self.N,self.L,self.PDE_Default_X(),self.BC_Default_X())
			Y=Grad_Post-Grad_Pre
			Denominator=jnp.inner(S,Y)
			Memory_DeltaW.Insert(S)
			Memory_DeltaGrad.Insert(Y)
			Memory_Ro.Insert(Denominator)
			Cost_Hist+=[Cost_Post]
			SV+=S*Grad_Pre
			Cost_Pre=Cost_Post
			Grad_Pre=np.copy(Grad_Post)
			Iters+=1
			if (np.abs(Alpha)<AlphaTol):
				Consecutive_Still+=1
			else:
				Consecutive_Still=0
		return {'Cost_History': Cost_Hist,'SV': SV}


	def Learn(self,ADAM_Steps,ADAM_Batch,SecondOrder_MaxSteps,Number_Test_Points,ADAM_Params=None,SecondOrder_Params=None):

		""" Learning Execution """

		Test_Residuals=Sample_Interior(self.Domain,Number_Test_Points['Internal'])
		Test_Boundary_Lists=Sample_Boundary(self.Domain,Number_Test_Points['Boundary'])
		Test_Dirichlet_Lists,Test_Dirichlet_Values,Test_Neumann_Lists,Test_Neumann_Values,Test_Periodic_Lists,Test_Periodic_Lower_Points,Test_Periodic_Upper_Points=Set_Boundary_Points_And_Values(Test_Boundary_Lists,self.Boundary_Labels,self.Exact_Boundary_Dirichlet,self.Exact_Boundary_Neumann)
		Test_Boundary_Spots=sum([Test_Boundary_Lists[i].shape[1] for i in range(len(Test_Boundary_Lists))])
		XR_Test={'Residual_Points': Test_Residuals, 'Number_Residuals': Number_Test_Points['Internal']}
		XB_Test={'Dirichlet_Lists': Test_Dirichlet_Lists,'Dirichlet_Values': Test_Dirichlet_Values,'Neumann_Lists': Test_Neumann_Lists,'Neumann_Values': Test_Neumann_Values,'Periodic_Lower_Points': Test_Periodic_Lower_Points,'Periodic_Upper_Points': Test_Periodic_Upper_Points,'Number_Boundary_Spots': Test_Boundary_Spots}
		Residuals_Count_List=[self.Number_Residuals]
		Learning_Iters_List=[]
		Cost_History=[]
		Continue=True
		print("Saving Initial Settings ...")
		W_Pre=np.copy(self.Weights_On)
		A_Pre=np.copy(self.A)
		M_Pre=np.copy(self.Mask)
		Rows_Pre,Cum_Pre=Globals()
		Penalty_Pre=self.Cost(self.Weights_On,self.A,self.N,np.ones(Number_Test_Points['Internal']+Test_Boundary_Spots),XR_Test,XB_Test)
		while (Continue):
			print("Starting Learning For Hidden Layer %d ..." %(self.Hidden_Layers))
			SV=np.zeros_like(self.Weights_On)
			print("ADAM Progressing ...")
			Out_First=self.ADAM(ADAM_Steps,ADAM_Batch,ADAM_Params)
			Cost_History+=Out_First['Cost_History'][:-1]
			SV+=Out_First['SV']
			print("L-BFGS-Plus Progressing ...")
			Out_Second=self.LBFGS_Plus(SecondOrder_MaxSteps,SecondOrder_Params)
			Cost_History+=Out_Second['Cost_History']
			SV+=Out_Second['SV']
			print("Pruning ...")
			SV=np.multiply(SV,np.divide(self.Weights_On,(self.Weights_On-W_Pre),out=np.zeros_like(SV),where=(np.abs(self.Weights_On-W_Pre)>self.NAE_Options['DiffTol'])),out=np.copy(self.Weights_On),where=(np.abs(SV>EpsMachine)))
			self.Pruning(SV)
			print("Retrain (L-BFGS-Plus) ...")
			Out_Retrain=self.LBFGS_Plus(SecondOrder_MaxSteps//10,SecondOrder_Params,True)
			Cost_History+=Out_Retrain['Cost_History']
			Learning_Iters_List+=[len(Cost_History)]
			print("Evaluating & Comparing ...")
			Residuals_Count_List+=[self.Number_Residuals]
			Penalty_Post=self.Cost(self.Weights_On,self.A,self.N,np.ones(Number_Test_Points['Internal']+Test_Boundary_Spots),XR_Test,XB_Test)
			if (Penalty_Pre<Penalty_Post):
				print("Previous Performance Was Better: Exiting ...")
				self.Weights_On=W_Pre
				self.A=A_Pre
				self.Mask=M_Pre
				if (self.Hidden_Layers!=1):
					self.Hidden_Layers-=1
				return Cost_History,Learning_Iters_List,Residuals_Count_List,Rows_Pre,Cum_Pre
			elif (self.Hidden_Layers==self.NAE_Options['Max_Hidden_Layers']):
				print("Last Performance Is Better But Cannot Add More Hidden Layers: Exiting ...")
				Rows_Post,Cum_Post=Globals()
				return Cost_History,Learning_Iters_List,Residuals_Count_List,Rows_Post,Cum_Post
			else:
				print("Last Performance Is Better: Continuing ...")
				W_Pre=np.copy(self.Weights_On)
				A_Pre=np.copy(self.A)
				M_Pre=np.copy(self.Mask)
				Rows_Pre,Cum_Pre=Globals()
				Penalty_Pre=self.Cost(self.Weights_On,self.A,self.N,np.ones(Number_Test_Points['Internal']+Test_Boundary_Spots),XR_Test,XB_Test)
				self.Growing()
