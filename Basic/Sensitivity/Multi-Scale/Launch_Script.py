#! /usr/bin/python3

from Basic.PINN_Resolutors import *


""" Launch Script [To Be Launched Once For Each Test] [To Be Tuned By Hand For Every Launch] -> Basic PINN Sensitivity Analysis (Multi-Scale)

	Carried Basing On Results -> Basic PINN Sensitivity Analysis (Single-Scale)

	Problem: Scalar 1D Poisson With Homogeneous Dirichlet Conditions In Domain [-1,1]

	Launch To Create PINNs To Approximate:
	- Multi-Scale Frequency Solution

	Trial Activation Function:
	- Tanh (Hyperbolic Tangent)

	Trial Architecture Parameters:
	- Hidden_Layers -> [1,2]
	- Neurons_Per_Layer -> [40,80]

	Trial Uniform Residuals:
	- [250,2500]

	Three Models Trained With [ADAM,L-BFGS] (10000 ADAM Steps (Default Settings), 50000 Maximum L-BFGS Steps (Default Settings Except Memory -> 100) [Both Full Batch]) For Each Combination -> [Residuals-Architecture-Activation]
	Each Model -> Saved In Proper Folder As Dictionary Containing:
	- History Cost Function
	- Final Relative L2 Error
	- Elapsed Learning Time
	- Error Discrete Fourier Transform
	- Network & Solution Evaluations
	- Solution String """


Coeffs={'VeryLow': 15,'Low': 3,'Medium': 3/2,'High': 1,'VeryHigh': 3/5}

def F_VeryLow(X):
	return ((-1*jnp.pi**2)*jnp.sin(1*jnp.pi*X))
def Sol_VeryLow(X):
	return (jnp.sin(1*jnp.pi*X))
String_VeryLow='sin(1*pi*x)'

def F_Low(X):
	return ((-25*jnp.pi**2)*jnp.sin(5*jnp.pi*X))
def Sol_Low(X):
	return (jnp.sin(5*jnp.pi*X))
String_Low='sin(5*pi*x)'

def F_Medium(X):
	return (-100*jnp.pi**2)*jnp.sin(10*jnp.pi*X)
def Sol_Medium(X):
	return (jnp.sin(10*jnp.pi*X))
String_Medium='sin(10*pi*x)'

def F_High(X):
	return (-225*jnp.pi**2)*jnp.sin(15*jnp.pi*X)
def Sol_High(X):
	return (jnp.sin(15*jnp.pi*X))
String_High='sin(15*pi*x)'

def F_VeryHigh(X):
	return (-625*jnp.pi**2)*jnp.sin(25*jnp.pi*X)
def Sol_VeryHigh(X):
	return (jnp.sin(25*jnp.pi*X))
String_VeryHigh='sin(25*pi*x)'

def F_Multi(X):
	return Coeffs['VeryLow']*F_VeryLow(X)+Coeffs['Low']*F_Low(X)+Coeffs['Medium']*F_Medium(X)+Coeffs['High']*F_High(X)+Coeffs['VeryHigh']*F_VeryHigh(X)
def S_Multi(X):
	return Coeffs['VeryLow']*Sol_VeryLow(X)+Coeffs['Low']*Sol_Low(X)+Coeffs['Medium']*Sol_Medium(X)+Coeffs['High']*Sol_High(X)+Coeffs['VeryHigh']*Sol_VeryHigh(X)
String_Multi=str(Coeffs['VeryLow'])+'*'+String_VeryLow+'+'+str(Coeffs['Low'])+'*'+String_Low+'+'+str(Coeffs['Medium'])+'*'+String_Medium+'+'+str(Coeffs['High'])+'*'+String_High+'+'+str(Coeffs['VeryHigh'])+'*'+String_VeryHigh

def G(X):
	return jnp.zeros_like(X)

Test=1
NAttempts=3
Number_Residuals=[250,2500]
ADAM_BatchFraction=[1.0,1.0]
ADAM_Steps=10000
LBFGS_MaxSteps=50000
Limits=np.array([[-1.0,1.0]])
Points=np.linspace(-1.0,1.0,20001)
Dx=Points[1]-Points[0]
Number_Boundary_Points=[[1,1]]
Boundary_Labels=2*['Dirichlet']
Hidden_Layers=[1,2]
Neurons_Per_Layer=[40,80]
Initialization='Uniform'
Activations={'Tanh': jnp.tanh}
ADAM_Parameters=None
LBFGS_Parameters={'Memory': 100,'GradTol': 1e-6,'AlphaTol': 1e-6,'StillTol': 10,'AlphaZero': 10.0,'C': 0.5,'T': 0.5,'Eps': 1e-8}


for c,NR in enumerate(Number_Residuals):
	for SigmaName,Sigma in Activations.items():
		for HL in Hidden_Layers:
			for NPL in Neurons_Per_Layer:
					for i in range(NAttempts):
						Architecture={'Input_Dimension': 1,'Hidden_Layers': HL,'Neurons_Per_Layer': HL*[NPL],'Activation': Sigma,'Initialization': Initialization}
						Domain={'Dimension': 1,'Limits': Limits,'Number_Residuals': NR,'Number_Boundary_Points': Number_Boundary_Points,'Boundary_Labels': Boundary_Labels}
						Data={'Source': F_Multi,'Exact_Dirichlet': G,'Exact_Neumann': G}
						Solver=Resolutor_Basic[Poisson_Scalar_Basic](Architecture,Domain,Data)
						Start=time.perf_counter()
						History=Solver.Learn(ADAM_Steps,ADAM_BatchFraction[c],LBFGS_MaxSteps,ADAM_Params=ADAM_Parameters,LBFGS_Params=LBFGS_Parameters)
						End=time.perf_counter()
						Elapsed=End-Start
						Network_Eval=Solver.Network_Multiple(Points[None,:],Solver.Architecture['W'])[0,:]
						Solution_Eval=S_Multi(Points)
						Error=Solution_Eval-Network_Eval
						Relative_L2_Error=np.linalg.norm(Error)/np.linalg.norm(Solution_Eval)
						[DFT_Error,Freqs]=[np.fft.fftshift(np.fft.fft(Error)),np.fft.fftshift(np.fft.fftfreq(len(Error),d=Dx))]
						Dictionary={'Time': Elapsed,'History': History,'Relative_L2_Error': Relative_L2_Error,'DFT_Error': DFT_Error,'DFT_Freqs': Freqs,'Solution': String_Multi,'Network_Eval': Network_Eval,'Solution_Eval': Solution_Eval}
						Name='Models_'+SigmaName+'/'+'Test_'+str(Test)+'/'+'Model_Multi_'+str(HL)+'HL_'+str(NPL)+'NPL_'+str(NR)+'NR_'+str(i+1)
						File=open(Name,'wb')
						dill.dump(Dictionary,File)
						File.close()
						Solver.Print_Cost()
						print('Model Saved Successfully')
