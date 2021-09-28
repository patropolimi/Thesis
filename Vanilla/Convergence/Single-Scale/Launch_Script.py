#! /usr/bin/python3

from PINN_Resolutors import *


""" Main Script [To Be Launched Once For Each Test] [To Be Tuned By Hand For Every Launch] -> PINN Convergence Analysis

	Problem: Scalar 1D Poisson With Homogeneous Dirichlet Conditions In Domain [-1,1]

    Objective -> Evaluate Convergence Of PINN With Respect To Number Of Residuals [Fixed Architecture]

    Three Models Trained With ADAM-BFGS (10000 ADAM Steps, 10000 Maximum BFGS Steps [Both Default Settings]) For Each Choice Of Residual Number
	Each Model -> Saved In Proper Folder As Dictionary Containing [Evaluations Made On Default Uniform Linspacing]:
	- Final Weights
	- Final Relative L2 Error
	- Batch Size
	- Elapsed Learning Time
	- History Cost Function
	- Discrete Fourier Transform Vector (Along With Frequencies)
	- Solution String
	- Network Evaluations
	- Solution Evaluations
    - Activation
    - Hidden Layers
    - Neurons Per Layer """


def F(X):
	return ((-225*jnp.pi**2)*jnp.sin(15*jnp.pi*X))
def S(X):
	return (jnp.sin(15*jnp.pi*X))
SolutionString='sin(15*pi*x)'

def G(X):
	return jnp.zeros_like(X)

Number_Residuals=[10,20,40,80,160,320]
ADAM_Batch=[10,20,40,80,160,320]
ADAM_Steps=10000
BFGS_MaxSteps=10000
NAttempts=3
Domain=np.array([[-1.0,1.0]])
Points=np.linspace(-1.0,1.0,2001)
Dx=Points[1]-Points[0]
Number_Boundary_Points=[[1,1]]
Boundary_Labels=2*['Dirichlet']
Hidden_Layers=1
Neurons_Per_Layer=50
Sigma=jnp.tanh
SigmaName='Tanh'
Test_ID=1


for c,NR in enumerate(Number_Residuals):
	for i in range(NAttempts):
		Solver=Resolutor_ADAM_BFGS[Poisson_Scalar_Basic](1,Hidden_Layers,Neurons_Per_Layer,Sigma,Domain,NR,Number_Boundary_Points,Boundary_Labels,F,G,G)
		Start=time.perf_counter()
		History=Solver.Learn(ADAM_Steps,ADAM_Batch[c],BFGS_MaxSteps)
		End=time.perf_counter()
		Elapsed=End-Start
		W=copy.deepcopy(Solver.Weights)
		Network_Eval=Solver.Network_Multiple(Points[None,:],W)[0,:]
		Solution_Eval=S(Points)
		Error=Solution_Eval-Network_Eval
		Relative_L2_Error=np.linalg.norm(Error)/np.linalg.norm(Solution_Eval)
		[DFT_Error,Freqs]=[np.fft.fft(Error),np.fft.fftfreq(len(Error),d=Dx)]
		[DFT_Error,Freqs]=[np.fft.fftshift(DFT_Error),np.fft.fftshift(Freqs)]
		Dictionary={'W': W,'Time': Elapsed,'History': History,'Relative_L2_Error': Relative_L2_Error,'DFT_Error': DFT_Error,'Freqs_DFT': Freqs,'Hidden_Layers': Hidden_Layers,
					'Neurons_Per_Layer': Neurons_Per_Layer,'Solution': SolutionString,'Network_Eval': Network_Eval,'Solution_Eval': Solution_Eval,'Batch_Size': ADAM_Batch[c],'Activation': SigmaName}
		Name='Test_'+str(Test_ID)+'/Model_'+str(NR)+'NR_'+str(i+1)
		File=open(Name,'wb')
		dill.dump(Dictionary,File)
		File.close()
		Solver.Print_Cost()
		print("Model Saved Successfully")
