#! /usr/bin/python3

from Basic.PINN_Resolutors import *


""" Launch Script For Advection 1D [To Be Launched Once For Each Test] [To Be Tuned By Hand For Every Launch] """


def F(X):
	return jnp.sum(jnp.zeros_like(X),axis=0)[None,:]
def S(X):
	return (jnp.exp(-((X[1,:]-X[0,:])/(0.1))**2))[None,:]
SolutionString='1*exp(-((x-t)/(0.1))**2)'

def G(X):
	return jnp.where(jnp.abs(X[0,:])<EpsMachine,(jnp.exp(-((X[1,:])/(0.1))**2)),jnp.sum(jnp.zeros_like(X),axis=0))[None,:]

Test=1
NAttempts=3
Number_Residuals=[2500]
ADAM_BatchFraction=[1.0]
ADAM_Steps=10000
LBFGS_MaxSteps=50000
Limits=np.array([[0.0,1.0],[-1.0,1.0]])
NT=1001
NX=2001
PT=np.linspace(0.0,1.0,NT)
PX=np.linspace(-1.0,1.0,NX)
TV=PT[None,:]
Points_2D=[]
for i in range(NX):
	XV=np.array(NT*[PX[i]])[None,:]
	Points_2D+=[np.concatenate((TV,XV),axis=0)]
Points_2D=np.concatenate(Points_2D,axis=1)
Number_Boundary_Points=[[100,0],[0,0]]
Boundary_Labels=['Dirichlet','None','None','None']
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
						Data={'Source': F,'Exact_Dirichlet': G,'Exact_Neumann': G,'U': 1.0,'G': 0.0}
						Solver=Resolutor_Basic[Advection_Scalar_Basic](Architecture,Domain,Data)
						Start=time.perf_counter()
						History=Solver.Learn(ADAM_Steps,ADAM_BatchFraction[c],LBFGS_MaxSteps,ADAM_Params=ADAM_Parameters,LBFGS_Params=LBFGS_Parameters)
						End=time.perf_counter()
						Elapsed=End-Start
						Network_Eval=Solver.Network_Multiple(Points_2D,Solver.Architecture['W'])
						Solution_Eval=S(Points_2D)
						Error=Solution_Eval-Network_Eval
						Relative_L2_Error=np.linalg.norm(Error)/np.linalg.norm(Solution_Eval)
						Dictionary={'Time': Elapsed,'History': History,'Relative_L2_Error': Relative_L2_Error,'Solution': SolutionString,'Network_Eval': Network_Eval,'Solution_Eval': Solution_Eval}
						Name='Models_'+SigmaName+'/'+'Test_'+str(Test)+'/'+'Model_'+str(HL)+'HL_'+str(NPL)+'NPL_'+str(NR)+'NR_'+str(i+1)
						File=open(Name,'wb')
						dill.dump(Dictionary,File)
						File.close()
						Solver.Print_Cost()
						print('Model Saved Successfully')
