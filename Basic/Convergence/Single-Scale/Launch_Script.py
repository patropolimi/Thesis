#! /usr/bin/python3

from Basic.PINN_Resolutors import *


""" Launch Script [To Be Launched Once For Each Test] [To Be Tuned By Hand For Every Launch] -> Basic PINN Convergence Analysis (Single-Scale)

	Carried Basing On Results -> Basic PINN Sensitivity Analysis

	Problem: Scalar 1D Poisson With Homogeneous Dirichlet Conditions In Domain [-1,1]

    Objective -> Evaluate Convergence Of PINN With Respect To Number Of Residuals

	Three Models Trained With [ADAM,L-BFGS] (10000 ADAM Steps (Default Settings), 50000 Maximum L-BFGS Steps (Default Settings Except Memory -> 100) [Both Full Batch]) For Each Combination Of [Residuals]
	Each Model -> Saved In Proper Folder As Dictionary Containing:
	- History Cost Function
	- Final Relative L2 Error
	- Elapsed Learning Time
	- Error Discrete Fourier Transform
	- Network & Solution Evaluations
	- Solution String """


def F(X):
	return ((-25*jnp.pi**2)*jnp.sin(5*jnp.pi*X))
def S(X):
	return (jnp.sin(5*jnp.pi*X))
SolutionString='sin(5*pi*x)'

def G(X):
	return jnp.zeros_like(X)

Test=1
NAttempts=3
Number_Residuals=[100,200,400,800,1600,3200]
ADAM_BatchFraction=[1.0,1.0,1.0,1.0,1.0,1.0]
ADAM_Steps=10000
LBFGS_MaxSteps=50000
Limits=np.array([[-1.0,1.0]])
Points=np.linspace(-1.0,1.0,20001)
Dx=Points[1]-Points[0]
Number_Boundary_Points=[[1,1]]
Boundary_Labels=2*['Dirichlet']
Hidden_Layers=1
Neurons_Per_Layer=40
Initialization='Uniform'
Sigma=jnp.tanh
SigmaName='Tanh'
ADAM_Parameters=None
LBFGS_Parameters={'Memory': 100,'GradTol': 1e-6,'AlphaTol': 1e-6,'StillTol': 10,'AlphaZero': 10.0,'C': 0.5,'T': 0.5,'Eps': 1e-8}


for c,NR in enumerate(Number_Residuals):
	for i in range(NAttempts):
		Architecture={'Input_Dimension': 1,'Hidden_Layers': Hidden_Layers,'Neurons_Per_Layer': Hidden_Layers*[Neurons_Per_Layer],'Activation': Sigma,'Initialization': Initialization}
		Domain={'Dimension': 1,'Limits': Limits,'Number_Residuals': NR,'Number_Boundary_Points': Number_Boundary_Points,'Boundary_Labels': Boundary_Labels}
		Data={'Source': F,'Exact_Dirichlet': G,'Exact_Neumann': G}
		Solver=Resolutor_Basic[Poisson_Scalar_Basic](Architecture,Domain,Data)
		Start=time.perf_counter()
		History=Solver.Learn(ADAM_Steps,ADAM_BatchFraction[c],LBFGS_MaxSteps,ADAM_Params=ADAM_Parameters,LBFGS_Params=LBFGS_Parameters)
		End=time.perf_counter()
		Elapsed=End-Start
		Network_Eval=Solver.Network_Multiple(Points[None,:],Solver.Architecture['W'])[0,:]
		Solution_Eval=S(Points)
		Error=Solution_Eval-Network_Eval
		Relative_L2_Error=np.linalg.norm(Error)/np.linalg.norm(Solution_Eval)
		[DFT_Error,Freqs]=[np.fft.fftshift(np.fft.fft(Error)),np.fft.fftshift(np.fft.fftfreq(len(Error),d=Dx))]
		Dictionary={'Time': Elapsed,'History': History,'Relative_L2_Error': Relative_L2_Error,'DFT_Error': DFT_Error,'DFT_Freqs': Freqs,'Solution': SolutionString,'Network_Eval': Network_Eval,'Solution_Eval': Solution_Eval,'Hidden_Layers': Hidden_Layers,'Neurons_Per_Layer': Neurons_Per_Layer}
		Name='Models_'+SigmaName+'/'+'Test_'+str(Test)+'/'+'Model_'+str(NR)+'NR_'+str(i+1)
		File=open(Name,'wb')
		dill.dump(Dictionary,File)
		File.close()
		Solver.Print_Cost()
		print('Model Saved Successfully')
