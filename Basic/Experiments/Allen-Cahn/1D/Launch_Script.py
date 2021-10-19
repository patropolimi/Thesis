#! /usr/bin/python3

from Basic.PINN_Resolutors import *


""" Launch Script For Allen-Cahn 1D [To Be Launched Once For Each Test] [To Be Tuned By Hand For Every Launch] """


def F(X):
	return jnp.sum(jnp.zeros_like(X),axis=0)[None,:]
File=open('High_Fidelity'+'/'+'Solution','rb')
Solution_Data=dill.load(File)
File.close()

def G(X):
	return ((X[1,:]**2)*jnp.cos(jnp.pi*X[1,:]))[None,:]

Test=1
NAttempts=3
Number_Residuals=[500]
ADAM_BatchFraction=[1.0]
ADAM_Steps=10000
LBFGS_MaxSteps=50000
Limits=np.array([[0.0,1.0],[-1.0,1.0]])
PT=Solution_Data['T']
PX=Solution_Data['X']
NT=PT.shape[0]
NX=PX.shape[0]
TV=PT[None,:]
Points_2D=[]
for i in range(NX):
	XV=np.array(NT*[PX[i]])[None,:]
	Points_2D+=[np.concatenate((TV,XV),axis=0)]
Points_2D=np.concatenate(Points_2D,axis=1)
Number_Boundary_Points=[[200,0],[100,100]]
Boundary_Labels=['Dirichlet','None','Periodic','Periodic']
Hidden_Layers=[1]
Neurons_Per_Layer=[150]
Initialization='Uniform'
Activations={'Tanh': jnp.tanh}
ADAM_Parameters=None
LBFGS_Parameters={'Memory': 100,'GradTol': 1e-6,'AlphaTol': 1e-6,'StillTol': 10,'AlphaZero': 10.0,'C': 0.5,'T': 0.5,'Eps': 1e-8}


for c,NR in enumerate(Number_Residuals):
	for SigmaName,Sigma in Activations.items():
		for HL in Hidden_Layers:
			for NPL in Neurons_Per_Layer:
					for i in range(NAttempts):
						Architecture={'Input_Dimension': 2,'Hidden_Layers': HL,'Neurons_Per_Layer': HL*[NPL],'Activation': Sigma,'Initialization': Initialization}
						Domain={'Dimension': 2,'Limits': Limits,'Number_Residuals': NR,'Number_Boundary_Points': Number_Boundary_Points,'Boundary_Labels': Boundary_Labels}
						Data={'Source': F,'Exact_Dirichlet': G,'Exact_Neumann': G,'Gamma1': 1e-4,'Gamma2': 5.0}
						Solver=Resolutor_Basic[Allen_Cahn_Scalar_Basic](Architecture,Domain,Data)
						Start=time.perf_counter()
						History=Solver.Learn(ADAM_Steps,ADAM_BatchFraction[c],LBFGS_MaxSteps,ADAM_Params=ADAM_Parameters,LBFGS_Params=LBFGS_Parameters)
						End=time.perf_counter()
						Elapsed=End-Start
						Network_Eval=Solver.Network_Multiple(Points_2D,Solver.Architecture['W'])
						Solution_Eval=Solution_Data['Values']
						Error=Solution_Eval-Network_Eval
						Relative_L2_Error=np.linalg.norm(Error)/np.linalg.norm(Solution_Eval)
						Dictionary={'Time': Elapsed,'History': History,'Relative_L2_Error': Relative_L2_Error,'Network_Eval': Network_Eval,'Solution_Eval': Solution_Eval}
						Name='Models_'+SigmaName+'/'+'Test_'+str(Test)+'/'+'Model_'+str(HL)+'HL_'+str(NPL)+'NPL_'+str(NR)+'NR_'+str(i+1)
						File=open(Name,'wb')
						dill.dump(Dictionary,File)
						File.close()
						Solver.Print_Cost()
						print('Model Saved Successfully')
