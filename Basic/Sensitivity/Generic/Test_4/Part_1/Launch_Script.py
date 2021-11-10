#! /usr/bin/python3

from Basic.PINN_Resolutors import *


""" Test 4 [Part 1] Launch Script -> Basic PINN Generic Sensitivity Analysis"""


def H(X):
	return (jax.nn.relu(2*X)-2*jax.nn.relu(2*X-1)+jax.nn.relu(2*X-2))
NHats=1
def S(X):
	Y=H(X)
	for i in range(NHats-1):
		Y=H(Y)
	return Y
def SingleS(X):
	return jnp.sum(S(X))
String=str(2**(NHats-1))+'Teeth_Saw_[0,1]'
def F(X):
	return (jax.vmap(jax.grad(SingleS),in_axes=1)(X))[None,:]
def G(X):
	return jnp.zeros_like(X)
NAttempts=3
Number_Residuals=[20,40,80,160]
ADAM_Steps=10000
LBFGS_MaxSteps=20000
Limits=np.array([[0.0,1.0]])
Points=np.linspace(0.0,1.0,1001)
Number_Boundary_Points=[[1,0]]
Boundary_Labels=['Dirichlet','None']
Hidden_Layers=[1,2]
Neurons_Per_Layer=[25,50,100]
Initialization='Uniform'
Activations={'Tanh': jnp.tanh,'Relu': jax.nn.relu}
Problem=ODE_Scalar_Basic
Parameters=None
for NR in Number_Residuals:
	for SigmaName,Sigma in Activations.items():
		for HL in Hidden_Layers:
			for NPL in Neurons_Per_Layer:
				for i in range(NAttempts):
					Architecture={'Input_Dimension': 1,'Hidden_Layers': HL,'Neurons_Per_Layer': HL*[NPL],'Activation': Sigma,'Initialization': Initialization}
					Domain={'Dimension': 1,'Limits': Limits,'Number_Residuals': NR,'Number_Boundary_Points': Number_Boundary_Points,'Boundary_Labels': Boundary_Labels}
					Data={'Source': F,'Exact_Dirichlet': G,'Exact_Neumann': None}
					Solver=Resolutor_Basic[Problem](Architecture,Domain,Data)
					Start=time.perf_counter()
					History=Solver.Learn(ADAM_Steps,LBFGS_MaxSteps,Parameters)
					End=time.perf_counter()
					Elapsed=End-Start
					Network_Eval=Solver.Network_Multiple(Points[None,:],Solver.Architecture['W'])[0,:]
					Solution_Eval=S(Points)
					Error=Solution_Eval-Network_Eval
					Relative_L2_Error=np.linalg.norm(Error)/np.linalg.norm(Solution_Eval)
					Dictionary={'Time': Elapsed,'History': History,'Relative_L2_Error': Relative_L2_Error,'Solution': String,'Network_Eval': Network_Eval,'Solution_Eval': Solution_Eval,'Points_Eval': Points}
					Name=SigmaName+'_Model_'+str(HL)+'HL_'+str(NPL)+'NPL_'+str(NR)+'NR_'+str(i+1)
					File=open(Name,'wb')
					dill.dump(Dictionary,File)
					File.close()
					Solver.Print_Cost()
					print('Model Saved Successfully')
