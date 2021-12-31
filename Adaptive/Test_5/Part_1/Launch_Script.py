#! /usr/bin/python3

from Adaptive.PINN_Resolutors import *


""" Test 5 [Part 2] Launch Script -> Adaptive PINN Generic Experiments """


def F(X):
	return (jnp.sum(jnp.zeros_like(X),axis=0))[None,:]
def S(X):
	return (1.0-jnp.tanh((1.0/5e-3)*(X[1,:]-X[0,:])))[None,:]
String='1-tanh(200*(x-t))'
def G(X):
	return (1.0-jnp.tanh((1.0/5e-3)*(X[1,:]-X[0,:])))[None,:]
Attempts=3
Limits=np.array([[-1.0,1.0],[-1.0,1.0]])
Points_T=np.linspace(-1.0,1.0,2001)
Points_X=np.linspace(-1.0,1.0,2001)
NT=Points_T.shape[0]
NX=Points_X.shape[0]
TV=Points_T[None,:]
Points=[]
for i in range(NX):
	XV=np.array(NT*[Points_X[i]])[None,:]
	Points+=[np.concatenate((TV,XV),axis=0)]
Points=np.concatenate(Points,axis=1)
Number_Boundary_Points=[[200,0],[100,100]]
Boundary_Labels=['Dirichlet','None','Dirichlet','Dirichlet']
Initial_Number_Residuals=40
Initial_Hidden_Layers=1
Initial_Neurons_Per_Layer=10
Pool_Residuals_Size=20000
Max_Number_Residuals=5120
Max_Hidden_Layers=3
Min_Neurons_Per_Layer=10
Max_Neurons_Per_Layer=160
NoAction_Threshold=10
RAR_Threshold=1.05
GRW_Threshold=1000
Force_First_Iterations=10
Learning_Iterations_Multiplier=1
Initialization='Glorot_Uniform'
Activation=jnp.tanh
Problem=Burgers_Scalar_Adaptive
Parameters=None


Architecture={'Input_Dimension': 2,'Hidden_Layers': Initial_Hidden_Layers,'Neurons_Per_Layer': Initial_Hidden_Layers*[Initial_Neurons_Per_Layer],'Activation': Activation,'Initialization': Initialization}
Domain={'Dimension': 2,'Limits': Limits,'Number_Residuals': Initial_Number_Residuals,'Number_Boundary_Points': Number_Boundary_Points,'Boundary_Labels': Boundary_Labels}
Data={'Source': F,'Exact_Dirichlet': G,'Exact_Neumann': None,'Nu': 2.5e-3}
Adaptivity={'Pool_Residuals_Size': Pool_Residuals_Size,'Max_Number_Residuals': Max_Number_Residuals,'Max_Hidden_Layers': Max_Hidden_Layers,'Min_Neurons_Per_Layer': Min_Neurons_Per_Layer,'Max_Neurons_Per_Layer': Max_Neurons_Per_Layer,'NoAction_Threshold': NoAction_Threshold,'RAR_Threshold': RAR_Threshold,'GRW_Threshold': GRW_Threshold,'Force_First_Iterations': Force_First_Iterations,'Learning_Iterations_Multiplier': Learning_Iterations_Multiplier}
for i in range(Attempts):
	Solver=Resolutor_Adaptive[Problem](copy.deepcopy(Architecture),copy.deepcopy(Domain),copy.deepcopy(Data),copy.deepcopy(Adaptivity))
	Start=time.perf_counter()
	Saliency_History=Solver.Main(Parameters)
	End=time.perf_counter()
	Elapsed=End-Start
	Network_Eval=Solver.Network_Multiple(Points,Solver.Architecture['W'])
	Solution_Eval=S(Points)
	Error=Solution_Eval-Network_Eval
	Relative_L2_Error=np.linalg.norm(Error)/np.linalg.norm(Solution_Eval)
	Dictionary={'Time': Elapsed,'Saliency_History': Saliency_History,'Relative_L2_Error': Relative_L2_Error,'Solution': String,'Network_Eval': Network_Eval,'Solution_Eval': Solution_Eval,'Points_Abscissa': Points_T,'Points_Ordinate': Points_X,'Structure': Solver.Architecture['Neurons_Per_Layer'],'Final_Number_Residuals': Solver.Domain['Number_Residuals']}
	Name='Model_'+str(i+1)
	File=open(Name,'wb')
	dill.dump(Dictionary,File)
	File.close()
	Solver.Print_Cost()
	print('Model Saved Successfully')
