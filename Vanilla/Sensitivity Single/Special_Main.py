#! /usr/bin/python3

from PINN_Resolutors import *


""" Special Models Script [To Be Tuned By Hand For Every Launch] -> PINN Sensitivity Analysis

	Problem: Scalar 1D Poisson With Homogeneous Dirichlet Conditions In Domain [-1,1]

	Instructions:
	- Tune Models Settings Below
	- Script Saves Model Information In Extra Folder """

# Settings
def U(X):
	return (jnp.sin(15*jnp.pi*X))
def F(X):
	return (-225*jnp.pi**2)*U(X)
def G(X):
	return jnp.zeros_like(X)
Solution='sin(15*pi*x)'
Number_Residuals=200
ADAM_Steps=15000
ADAM_Batch=200
BFGS_MaxSteps=10000
Hidden_Layers=1
Neurons_Per_Layer=50
Domain=np.array([[-1.0,1.0]])
Number_Boundary_Points=[[1,1]]
Boundary_Labels=2*['Dirichlet']
Sigma=jnp.tanh
SigmaName='Tanh'
Points=np.linspace(-1.0,1.0,2001)
dx=Points[1]-Points[0]

# Execution
ID=int(input("Insert ID: "))
Solver=Resolutor_ADAM_BFGS[Poisson_Scalar_Basic](1,Hidden_Layers,Neurons_Per_Layer,Sigma,Domain,Number_Residuals,Number_Boundary_Points,Boundary_Labels,F,G,G)
Start=time.perf_counter()
History=Solver.Learn(ADAM_Steps,ADAM_Batch,BFGS_MaxSteps)
End=time.perf_counter()
Elapsed=End-Start
W=Solver.Weights
Network_Eval=Solver.Network_Multiple(Points[None,:],W)[0,:]
Solution_Eval=U(Points)
Error=Solution_Eval-Network_Eval
Relative_L2_Error=np.linalg.norm(Error)/np.linalg.norm(Solution_Eval)
[DFT_Error,Freqs]=[np.fft.fft(Error),np.fft.fftfreq(len(Error),d=dx)]
[DFT_Error,Freqs]=[np.fft.fftshift(DFT_Error),np.fft.fftshift(Freqs)]
Dictionary={'W': W,'Time': Elapsed,'History': History,'Relative_L2_Error': Relative_L2_Error,'DFT_Error': DFT_Error,'Freqs_DFT': Freqs,'Solution': Solution,'Activation': SigmaName,
			'Number_Residuals': Number_Residuals,'ADAM_Steps': ADAM_Steps,'ADAM_Batch': ADAM_Batch,'BFGS_MaxSteps': BFGS_MaxSteps,'Hidden_Layers': Hidden_Layers,'Neurons_Per_Layer': Neurons_Per_Layer}
Name='./Extra/Model_'+str(ID)
File=open(Name,'wb')
dill.dump(Dictionary,File)
File.close()
print("Model Saved Successfully")
