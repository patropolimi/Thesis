#! /usr/bin/python3

from PINN_Resolutors import *


""" Main Script [To Be Launched Once For Each Test] [To Be Tuned By Hand For Every Launch] -> PINN Sensitivity Analysis

	{In Each Test -> Change F_VeryLow, F_Low, F_Medium, F_High, F_VeryHigh + Relative Solutions & Strings + Residual Points + ADAM Batch}

	Problem: Scalar 1D Poisson With Homogeneous Dirichlet Conditions In Domain [-1,1]

	Launch To Create PINNs To Approximate:
	- Very Low Frequency Solution (F_VeryLow: Corresponding Source Term)
	- Low Frequency Solution (F_Low: Corresponding Source Term)
	- Medium Frequency Solution (F_Medium: Corresponding Source Term)
	- High Frequency Solution (F_High: Corresponding Source Term)
	- Very High Frequency Solution (F_VeryHigh: Corresponding Source Term)

	Trial Activation Functions:
	- ReLU (Rectified Linear Unit)
	- Tanh (Hyperbolic Tangent)
	- Sigmoid (Classic Sigmoid)

	Trial Architectures Parameters:
	- Hidden_Layers -> [1,2,3]
	- Neurons_Per_Layer -> [15,30,50]

	Three Models Trained With ADAM-BFGS (10000 ADAM Steps, 10000 Maximum BFGS Steps [Both Default Settings]) For Each (Residuals-Frequency-Architecture-Activation) Combination
	Each Model -> Saved In Proper Folder As Dictionary Containing [Evaluations Made On Default Uniform Linspacing]:
	- Final Weights
	- Final Relative L2 Error
	- Batch Size
	- Elapsed Learning Time
	- History Cost Function
	- Discrete Fourier Transform Vector (Along With Frequencies)
	- Solution String
	- Network Evaluations
	- Solution Evaluations """

C={'VeryLow': 2.0,'Low': 1.0,'Medium': 0.5,'High': 0.25,'VeryHigh': 0.125}

def F_VeryLow(X):
	return ((-0.25*jnp.pi**2)*jnp.cos(0.5*jnp.pi*X))
def Sol_VeryLow(X):
	return (jnp.cos(0.5*jnp.pi*X))
String_VeryLow='cos(0.5*pi*x)'

def F_Low(X):
	return ((-jnp.pi**2)*jnp.sin(jnp.pi*X))
def Sol_Low(X):
	return (jnp.sin(jnp.pi*X))
String_Low='sin(pi*x)'

def F_Medium(X):
	return (-25*jnp.pi**2)*jnp.sin(5*jnp.pi*X)
def Sol_Medium(X):
	return (jnp.sin(5*jnp.pi*X))
String_Medium='sin(5*pi*x)'

def F_High(X):
	return (-100*jnp.pi**2)*jnp.sin(10*jnp.pi*X)
def Sol_High(X):
	return (jnp.sin(10*jnp.pi*X))
String_High='sin(10*pi*x)'

def F_VeryHigh(X):
	return (-225*jnp.pi**2)*jnp.sin(15*jnp.pi*X)
def Sol_VeryHigh(X):
	return (jnp.sin(15*jnp.pi*X))
String_VeryHigh='sin(15*pi*x)'

def F(X):
	return C['VeryLow']*F_VeryLow(X)+C['Low']*F_Low(X)+C['Medium']*F_Medium(X)+C['High']*F_High(X)+C['VeryHigh']*F_VeryHigh(X)
def S(X):
	return C['VeryLow']*Sol_VeryLow(X)+C['Low']*Sol_Low(X)+C['Medium']*Sol_Medium(X)+C['High']*Sol_High(X)+C['VeryHigh']*Sol_VeryHigh(X)
String_Multiple=str(C['VeryLow'])+'*'+String_VeryLow+'+'+str(C['Low'])+'*'+String_Low+'+'+str(C['Medium'])+'*'+String_Medium+'+'+str(C['High'])+'*'+String_High+'+'+str(C['VeryHigh'])+'*'+String_VeryHigh

def G(X):
	return jnp.zeros_like(X)

Number_Residuals=[200]
ADAM_Batch=[200]
ADAM_Steps=1
BFGS_MaxSteps=1
NTests=3
Domain=np.array([[-1.0,1.0]])
Points=np.linspace(-1.0,1.0,2001)
Dx=Points[1]-Points[0]
Number_Boundary_Points=[[1,1]]
Boundary_Labels=2*['Dirichlet']
Hidden_Layers=[1,2,3]
Neurons_Per_Layer=[15,30,50]
Activations={'Tanh': jnp.tanh}
Test=1


for c,NR in enumerate(Number_Residuals):
	for SigmaName,Sigma in Activations.items():
		for HL in Hidden_Layers:
			for NPL in Neurons_Per_Layer:
					for i in range(NTests):
						Solver=Resolutor_ADAM_BFGS_Basic[Poisson_Scalar_Basic](1,HL,NPL,Sigma,Domain,NR,Number_Boundary_Points,Boundary_Labels,F,G,G)
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
						Dictionary={'W': W,'Time': Elapsed,'History': History,'Relative_L2_Error': Relative_L2_Error,'DFT_Error': DFT_Error,'Freqs_DFT': Freqs,
									'Solution': String_Multiple,'Network_Eval': Network_Eval,'Solution_Eval': Solution_Eval,'Batch_Size': ADAM_Batch[c]}
						Name='Models_'+SigmaName+'/'+'Test_'+str(Test)+'/'+'Model'+'_'+str(HL)+'HL_'+str(NPL)+'NPL_'+str(NR)+'NR_'+str(i+1)
						File=open(Name,'wb')
						dill.dump(Dictionary,File)
						File.close()
						Solver.Print_Cost()
						print("Model Saved Successfully")
