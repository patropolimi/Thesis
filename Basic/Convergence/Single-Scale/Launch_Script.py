#! /usr/bin/python3

from PINN_Resolutors_Basic import *


""" Launch Script [To Be Launched Once For Each Test] [To Be Tuned By Hand For Every Launch] -> Basic PINN Convergence Analysis (Single-Scale)

	Problem: Scalar 1D Poisson With Homogeneous Dirichlet Conditions In Domain [-1,1]

    Objective -> Evaluate Convergence Of PINN With Respect To Number Of Residuals


    Three Models Trained With ADAM-BFGS (10000 ADAM Steps, 10000 Maximum BFGS Steps [Both Default Settings]) For Each Choice [Residuals]
	Each Model -> Saved In Proper Folder As Dictionary Containing:
	- Final Weights
	- Final Relative L2 Error
	- Batch Size
	- Elapsed Learning Time
	- History Cost Function
	- Error Discrete Fourier Transform Vector & Frequencies
	- Solution String
	- Network & Solution Evaluations (Made On Default Uniform Linspacing)
	- Hidden Layers
    - Neurons Per Layer """


def F(X):
	return ((-jnp.pi**2)*jnp.sin(jnp.pi*X))
def S(X):
	return (jnp.sin(jnp.pi*X))
SolutionString='sin(pi*x)'

def G(X):
	return jnp.zeros_like(X)

Test=1
NAttempts=3
Number_Residuals=[100,200,400,800,1600]
ADAM_Batch=[100,200,400,800,1600]
ADAM_Steps=10000
BFGS_MaxSteps=10000
Domain=np.array([[-1.0,1.0]])
Points=np.linspace(-1.0,1.0,2001)
Dx=Points[1]-Points[0]
Number_Boundary_Points=[[1,1]]
Boundary_Labels=2*['Dirichlet']
Hidden_Layers=2
Neurons_Per_Layer=30
Sigma=jnp.tanh
SigmaName='Tanh'


for c,NR in enumerate(Number_Residuals):
	for i in range(NAttempts):
		Solver=Resolutor_ADAM_BFGS_Basic[Poisson_Scalar_Basic](1,Hidden_Layers,Neurons_Per_Layer,Sigma,Domain,NR,Number_Boundary_Points,Boundary_Labels,F,G,G)
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
					'Neurons_Per_Layer': Neurons_Per_Layer,'Solution': SolutionString,'Network_Eval': Network_Eval,'Solution_Eval': Solution_Eval,'Batch_Size': ADAM_Batch[c]}
		Name='Models_'+SigmaName+'/'+'Test_'+str(Test)+'/'+'Model_'+str(NR)+'NR_'+str(i+1)
		File=open(Name,'wb')
		dill.dump(Dictionary,File)
		File.close()
		Solver.Print_Cost()
		print("Model Saved Successfully")
