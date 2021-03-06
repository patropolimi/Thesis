#! /usr/bin/python3

from Basic.PINN_Resolutors import *


""" Test 2 [Part 1] Launch Script -> Basic PINN Multi-Scale Sensitivity Analysis """


def F_Low(X):
	return ((-4*jnp.pi**2)*jnp.sin(2*jnp.pi*X))
def Sol_Low(X):
	return (jnp.sin(2*jnp.pi*X))
String_Low='sin(2*pi*x)'
def F_High(X):
	return (-2500*jnp.pi**2)*jnp.sin(50*jnp.pi*X)
def Sol_High(X):
	return (jnp.sin(50*jnp.pi*X))
String_High='sin(50*pi*x)'
Coeffs={'Low': 1.0,'High': 0.1}
def F_Multi(X):
	return (Coeffs['Low']*F_Low(X)+Coeffs['High']*F_High(X))
def Sol_Multi(X):
	return (Coeffs['Low']*Sol_Low(X)+Coeffs['High']*Sol_High(X))
String_Multi=str(Coeffs['Low'])+'*'+String_Low+'+'+str(Coeffs['High'])+'*'+String_High
def G(X):
	return jnp.zeros_like(X)
NAttempts=3
Number_Residuals=[160,320,480]
ADAM_Steps=25000
LBFGS_MaxSteps=175000
Limits=np.array([[-1.0,1.0]])
Points=np.linspace(-1.0,1.0,2001)
Number_Boundary_Points=[[1,1]]
Boundary_Labels=2*['Dirichlet']
Hidden_Layers=[1,2]
Neurons_Per_Layer=[100,200,400]
Initialization='Glorot_Uniform'
Activations={'Tanh': jnp.tanh}
Problem=Poisson_Scalar_Basic
Parameters=None
for NR in Number_Residuals:
	for SigmaName,Sigma in Activations.items():
		for HL in Hidden_Layers:
			for NPL in Neurons_Per_Layer:
				for i in range(NAttempts):
					Architecture={'Input_Dimension': 1,'Hidden_Layers': HL,'Neurons_Per_Layer': HL*[NPL],'Activation': Sigma,'Initialization': Initialization}
					Domain={'Dimension': 1,'Limits': Limits,'Number_Residuals': NR,'Number_Boundary_Points': Number_Boundary_Points,'Boundary_Labels': Boundary_Labels}
					Data={'Source': F_Multi,'Exact_Dirichlet': G,'Exact_Neumann': None}
					Solver=Resolutor_Basic[Problem](Architecture,Domain,Data)
					Start=time.perf_counter()
					History=Solver.Learn(ADAM_Steps,LBFGS_MaxSteps,Parameters)
					End=time.perf_counter()
					Elapsed=End-Start
					Network_Eval=Solver.Network_Multiple(Points[None,:],Solver.Architecture['W'])[0,:]
					Solution_Eval=Sol_Multi(Points)
					Error=Solution_Eval-Network_Eval
					Relative_L2_Error=np.linalg.norm(Error)/np.linalg.norm(Solution_Eval)
					Dictionary={'Time': Elapsed,'History': History,'Relative_L2_Error': Relative_L2_Error,'Solution': String_Multi,'Network_Eval': Network_Eval,'Solution_Eval': Solution_Eval,'Points_Eval': Points}
					Name=SigmaName+'_Model_'+str(HL)+'HL_'+str(NPL)+'NPL_'+str(NR)+'NR_'+str(i+1)
					File=open(Name,'wb')
					dill.dump(Dictionary,File)
					File.close()
					Solver.Print_Cost()
					print('Model Saved Successfully')
