#! /usr/bin/python3

from Basic.PINN_Resolutors import *


""" Launch Script For Poisson 1D [To Be Launched Once For Each Test] [To Be Tuned By Hand For Every Launch] """


def F(X):
	return (-(jnp.pi**2)*(4*jnp.sin(2*jnp.pi*X)+250*jnp.sin(50*jnp.pi*X)))[None,:]
def S(X):
	return (jnp.sin(2*jnp.pi*X)+(1/10)*jnp.sin(50*jnp.pi*X))[None,:]
SolutionString='1*sin(2*pi*x)+0.1*sin(50*pi*x)'

def G(X):
	return jnp.sum(jnp.zeros_like(X),axis=0)[None,:]

Test=3
NAttempts=3
Number_Residuals=[500]
ADAM_BatchFraction=[1.0]
ADAM_Steps=10000
LBFGS_MaxSteps=350000
Limits=np.array([[-1.0,1.0]])
NX=2001
PX=np.linspace(-1.0,1.0,NX)
Number_Boundary_Points=[[1,1]]
Boundary_Labels=2*['Dirichlet']
Hidden_Layers=[1]
Neurons_Per_Layer=[350]
Initialization='Uniform'
Activations={'Tanh': jnp.tanh}
ADAM_Parameters=None
LBFGS_Parameters={'Memory': 100,'GradTol': 1e-6,'AlphaTol': 1e-6,'StillTol': 10,'AlphaZero': 10.0,'C': 0.5,'T': 0.5,'Eps': 1e-8}


for c,NR in enumerate(Number_Residuals):
	for SigmaName,Sigma in Activations.items():
		for HL in Hidden_Layers:
			for NPL in Neurons_Per_Layer:
					for i in range(NAttempts):
						Architecture={'Input_Dimension': 1,'Hidden_Layers': HL,'Neurons_Per_Layer': HL*[NPL],'Activation': Sigma,'Initialization': Initialization}
						Domain={'Dimension': 1,'Limits': Limits,'Number_Residuals': NR,'Number_Boundary_Points': Number_Boundary_Points,'Boundary_Labels': Boundary_Labels}
						Data={'Source': F,'Exact_Dirichlet': G,'Exact_Neumann': G}
						Solver=Resolutor_Basic[Poisson_Scalar_Basic](Architecture,Domain,Data)
						Start=time.perf_counter()
						History=Solver.Learn(ADAM_Steps,ADAM_BatchFraction[c],LBFGS_MaxSteps,ADAM_Params=ADAM_Parameters,LBFGS_Params=LBFGS_Parameters)
						End=time.perf_counter()
						Elapsed=End-Start
						Network_Eval=Solver.Network_Multiple(PX[None,:],Solver.Architecture['W'])
						Solution_Eval=S(PX[None,:])
						Error=Solution_Eval-Network_Eval
						Relative_L2_Error=np.linalg.norm(Error)/np.linalg.norm(Solution_Eval)
						Dictionary={'Time': Elapsed,'History': History,'Relative_L2_Error': Relative_L2_Error,'Solution': SolutionString,'Network_Eval': Network_Eval,'Solution_Eval': Solution_Eval}
						Name='Models_'+SigmaName+'/'+'Test_'+str(Test)+'/'+'Model_'+str(HL)+'HL_'+str(NPL)+'NPL_'+str(NR)+'NR_'+str(i+1)
						File=open(Name,'wb')
						dill.dump(Dictionary,File)
						File.close()
						Solver.Print_Cost()
						print('Model Saved Successfully')
