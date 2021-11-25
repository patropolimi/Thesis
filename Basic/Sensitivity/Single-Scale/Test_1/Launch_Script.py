#! /usr/bin/python3

from Basic.PINN_Resolutors import *


""" Test 1 Launch Script -> Basic PINN Single-Scale Sensitivity Analysis """


def F_Low(X):
	return ((-jnp.pi**2)*jnp.sin(jnp.pi*X))
def Sol_Low(X):
	return (jnp.sin(jnp.pi*X))
String_Low='sin(pi*x)'
def F_Medium(X):
	return (-25*jnp.pi**2)*jnp.sin(5*jnp.pi*X)
def Sol_Medium(X):
	return (jnp.sin(5*jnp.pi*X))
String_Medium='sin(5*pi*x)'
def F_Discrete(X):
	return (-100*jnp.pi**2)*jnp.sin(10*jnp.pi*X)
def Sol_Discrete(X):
	return (jnp.sin(10*jnp.pi*X))
String_Discrete='sin(10*pi*x)'
def F_High(X):
	return (-225*jnp.pi**2)*jnp.sin(15*jnp.pi*X)
def Sol_High(X):
	return (jnp.sin(15*jnp.pi*X))
String_High='sin(15*pi*x)'
def G(X):
	return jnp.zeros_like(X)
NAttempts=3
Number_Residuals=[80,320,1280]
ADAM_Steps=50000
LBFGS_MaxSteps=50000
Limits=np.array([[-1.0,1.0]])
Points=np.linspace(-1.0,1.0,2001)
Number_Boundary_Points=[[1,1]]
Boundary_Labels=2*['Dirichlet']
Hidden_Layers=[1,2]
Neurons_Per_Layer=[25,50,100]
Initialization='Glorot_Uniform'
Activations={'Tanh': jnp.tanh,'Relu': jax.nn.relu}
Sources={'Low': F_Low,'Medium': F_Medium,'Discrete': F_Discrete,'High': F_High}
Solution={'Low': Sol_Low,'Medium': Sol_Medium,'Discrete': Sol_Discrete,'High': Sol_High}
String={'Low': String_Low,'Medium': String_Medium,'Discrete': String_Discrete,'High': String_High}
Problem=Poisson_Scalar_Basic
Parameters=None
for NR in Number_Residuals:
	for SigmaName,Sigma in Activations.items():
		for HL in Hidden_Layers:
			for NPL in Neurons_Per_Layer:
				for Frequency,SRC in Sources.items():
					for i in range(NAttempts):
						Architecture={'Input_Dimension': 1,'Hidden_Layers': HL,'Neurons_Per_Layer': HL*[NPL],'Activation': Sigma,'Initialization': Initialization}
						Domain={'Dimension': 1,'Limits': Limits,'Number_Residuals': NR,'Number_Boundary_Points': Number_Boundary_Points,'Boundary_Labels': Boundary_Labels}
						Data={'Source': SRC,'Exact_Dirichlet': G,'Exact_Neumann': None}
						Solver=Resolutor_Basic[Problem](Architecture,Domain,Data)
						Start=time.perf_counter()
						History=Solver.Learn(ADAM_Steps,LBFGS_MaxSteps,Parameters)
						End=time.perf_counter()
						Elapsed=End-Start
						Network_Eval=Solver.Network_Multiple(Points[None,:],Solver.Architecture['W'])[0,:]
						Solution_Eval=Solution[Frequency](Points)
						Error=Solution_Eval-Network_Eval
						Relative_L2_Error=np.linalg.norm(Error)/np.linalg.norm(Solution_Eval)
						Dictionary={'Time': Elapsed,'History': History,'Relative_L2_Error': Relative_L2_Error,'Solution': String[Frequency],'Network_Eval': Network_Eval,'Solution_Eval': Solution_Eval,'Points_Eval': Points}
						Name=SigmaName+'_Model_'+Frequency+'_'+str(HL)+'HL_'+str(NPL)+'NPL_'+str(NR)+'NR_'+str(i+1)
						File=open(Name,'wb')
						dill.dump(Dictionary,File)
						File.close()
						Solver.Print_Cost()
						print('Model Saved Successfully')
