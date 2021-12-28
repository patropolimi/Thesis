#! /usr/bin/python3

from Basic.PINN_Resolutors import *


""" Test 6 Launch Script -> Basic PINN Generic Sensitivity Analysis """


def F(X):
	return (jnp.sum(jnp.zeros_like(X),axis=0))[None,:]
def S(X):
	return (1.0-jnp.tanh((1.0/5e-3)*(X[1,:]-X[0,:])))[None,:]
String='1-tanh(200*(x-t))'
def G(X):
	return (1.0-jnp.tanh((1.0/5e-3)*(X[1,:]-X[0,:])))[None,:]
NAttempts=3
Number_Residuals=[400,800]
ADAM_Steps=25000
LBFGS_MaxSteps=225000
Limits=np.array([[0.0,1.0],[0.0,1.0]])
Points_T=np.linspace(0.0,1.0,1001)
Points_X=np.linspace(0.0,1.0,1001)
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
Hidden_Layers=[1,2]
Neurons_Per_Layer=[25,50,100]
Initialization='Glorot_Uniform'
Activations={'Tanh': jnp.tanh,'Relu': jax.nn.relu}
Problem=Burgers_Scalar_Basic
Parameters=None
for NR in Number_Residuals:
	for SigmaName,Sigma in Activations.items():
		for HL in Hidden_Layers:
			for NPL in Neurons_Per_Layer:
				for i in range(NAttempts):
					Architecture={'Input_Dimension': 2,'Hidden_Layers': HL,'Neurons_Per_Layer': HL*[NPL],'Activation': Sigma,'Initialization': Initialization}
					Domain={'Dimension': 2,'Limits': Limits,'Number_Residuals': NR,'Number_Boundary_Points': Number_Boundary_Points,'Boundary_Labels': Boundary_Labels}
					Data={'Source': F,'Exact_Dirichlet': G,'Exact_Neumann': None,'Nu': 2.5e-3}
					Solver=Resolutor_Basic[Problem](Architecture,Domain,Data)
					Start=time.perf_counter()
					History=Solver.Learn(ADAM_Steps,LBFGS_MaxSteps,Parameters)
					End=time.perf_counter()
					Elapsed=End-Start
					Network_Eval=Solver.Network_Multiple(Points,Solver.Architecture['W'])
					Solution_Eval=S(Points)
					Error=Solution_Eval-Network_Eval
					Relative_L2_Error=np.linalg.norm(Error)/np.linalg.norm(Solution_Eval)
					Dictionary={'Time': Elapsed,'History': History,'Relative_L2_Error': Relative_L2_Error,'Solution': String,'Network_Eval': Network_Eval,'Solution_Eval': Solution_Eval,'Points_Abscissa': Points_T,'Points_Ordinate': Points_X}
					Name=SigmaName+'_Model_'+str(HL)+'HL_'+str(NPL)+'NPL_'+str(NR)+'NR_'+str(i+1)
					File=open(Name,'wb')
					dill.dump(Dictionary,File)
					File.close()
					Solver.Print_Cost()
					print('Model Saved Successfully')
