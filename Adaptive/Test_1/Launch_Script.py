#! /usr/bin/python3

from Adaptive.PINN_Resolutors import *


""" Test 1 Launch Script -> Adaptive PINN Generic Experiments """


def H(X):
	return (jax.nn.relu(2*X)-2*jax.nn.relu(2*X-1)+jax.nn.relu(2*X-2))
N=3
def S(X):
	Y=H(X)
	for i in range(N-1):
		Y=H(Y)
	return Y
String=str(2**(N-1))+'Teeth_Saw_[0,1]'
def F(X):
	return S(X)
def G(X):
	return jnp.zeros_like(X)
Attempts=3
Limits=np.array([[0.0,1.0]])
Points=np.linspace(0.0,1.0,10001)
Number_Boundary_Points=[[1,1]]
Boundary_Labels=2*['Dirichlet']
Initial_Number_Residuals=10
Initial_Hidden_Layers=1
Initial_Neurons_Per_Layer=10
Pool_Residuals_Size=10000
Max_Number_Residuals=5120
Max_Hidden_Layers=3
Min_Neurons_Per_Layer=10
Max_Neurons_Per_Layer=80
NoAction_Threshold=10
RAR_Threshold=1.05
GRW_Threshold=100
Force_First_Iterations=10
Learning_Iterations_Multiplier=1
Initialization='Glorot_Uniform'
Activation=jax.nn.relu
Problem=Identity_Scalar_Adaptive
Parameters=None


Architecture={'Input_Dimension': 1,'Hidden_Layers': Initial_Hidden_Layers,'Neurons_Per_Layer': Initial_Hidden_Layers*[Initial_Neurons_Per_Layer],'Activation': Activation,'Initialization': Initialization}
Domain={'Dimension': 1,'Limits': Limits,'Number_Residuals': Initial_Number_Residuals,'Number_Boundary_Points': Number_Boundary_Points,'Boundary_Labels': Boundary_Labels}
Data={'Source': F,'Exact_Dirichlet': G,'Exact_Neumann': None}
Adaptivity={'Pool_Residuals_Size': Pool_Residuals_Size,'Max_Number_Residuals': Max_Number_Residuals,'Max_Hidden_Layers': Max_Hidden_Layers,'Min_Neurons_Per_Layer': Min_Neurons_Per_Layer,'Max_Neurons_Per_Layer': Max_Neurons_Per_Layer,'NoAction_Threshold': NoAction_Threshold,'RAR_Threshold': RAR_Threshold,'GRW_Threshold': GRW_Threshold,'Force_First_Iterations': Force_First_Iterations,'Learning_Iterations_Multiplier': Learning_Iterations_Multiplier}
for i in range(Attempts):
	Solver=Resolutor_Adaptive[Problem](copy.deepcopy(Architecture),copy.deepcopy(Domain),copy.deepcopy(Data),copy.deepcopy(Adaptivity))
	Start=time.perf_counter()
	Saliency_History=Solver.Main(Parameters)
	End=time.perf_counter()
	Elapsed=End-Start
	Network_Eval=Solver.Network_Multiple(Points[None,:],Solver.Architecture['W'])[0,:]
	Solution_Eval=S(Points)
	Error=Solution_Eval-Network_Eval
	Relative_L2_Error=np.linalg.norm(Error)/np.linalg.norm(Solution_Eval)
	Dictionary={'Time': Elapsed,'Saliency_History': Saliency_History,'Relative_L2_Error': Relative_L2_Error,'Solution': String,'Network_Eval': Network_Eval,'Solution_Eval': Solution_Eval,'Points_Eval': Points,'Structure': Solver.Architecture['Neurons_Per_Layer'],'Final_Number_Residuals': Solver.Domain['Number_Residuals']}
	Name='Model_'+str(i+1)
	File=open(Name,'wb')
	dill.dump(Dictionary,File)
	File.close()
	Solver.Print_Cost()
	print('Model Saved Successfully')
