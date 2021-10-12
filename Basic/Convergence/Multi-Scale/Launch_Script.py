#! /usr/bin/python3

from Basic.PINN_Resolutors import *


""" Launch Script [To Be Launched Once For Each Test] [To Be Tuned By Hand For Every Launch] -> Basic PINN Convergence Analysis (Multi-Scale)

	Problem: Scalar 1D Poisson With Homogeneous Dirichlet Conditions In Domain [-1,1]

    Objective -> Evaluate Convergence Of PINN With Respect To Number Of Residuals

	Three Models Trained With [ADAM,L-BFGS] (10000 ADAM Steps, 20000 Maximum L-BFGS Steps [Both Default Settings & Full Batch]) For Each Combination Of [Residuals]
	Each Model -> Saved In Proper Folder As Dictionary Containing:
	- History Cost Function
	- Final Relative L2 Error
	- Elapsed Learning Time
	- Error Discrete Fourier Transform
	- Network & Solution Evaluations
	- Solution String """


Coeffs={'VeryLow': 5,'Low': 1,'Medium': 1/2,'High': 1/3,'VeryHigh': 1/5}

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
Number_Residuals=[100,200,400,800,1600,3200]
ADAM_BatchFraction=[1.0,1.0,1.0,1.0,1.0,1.0]
ADAM_Steps=10000
LBFGS_MaxSteps=20000
Domain=np.array([[-1.0,1.0]])
Points=np.linspace(-1.0,1.0,20001)
Dx=Points[1]-Points[0]
Number_Boundary_Points=[[1,1]]
Boundary_Labels=2*['Dirichlet']
Hidden_Layers=2
Neurons_Per_Layer=40
Initialization='Uniform'
Sigma=jnp.tanh
SigmaName='Tanh'


for c,NR in enumerate(Number_Residuals):
	for i in range(NAttempts):
		Architecture={'Input_Dimension': 1,'Hidden_Layers': HL,'Neurons_Per_Layer': HL*[NPL],'Activation': Sigma,'Initialization': Initialization}
		Domain={'Dimension': 1,'Limits': Limits,'Number_Residuals': NR,'Number_Boundary_Points': Number_Boundary_Points,'Boundary_Labels': Boundary_Labels}
		Data={'Source': F_Multi,'Exact_Dirichlet': G,'Exact_Neumann': G}
		Solver=Resolutor_Basic[Poisson_Scalar_Basic](Architecture,Domain,Data)
		Start=time.perf_counter()
		History=Solver.Learn(ADAM_Steps,ADAM_BatchFraction[c],LBFGS_MaxSteps)
		End=time.perf_counter()
		Elapsed=End-Start
		Network_Eval=Solver.Network_Multiple(Points[None,:],Solver.Architecture['W'])[0,:]
		Solution_Eval=S_Multi(Points)
		Error=Solution_Eval-Network_Eval
		Relative_L2_Error=np.linalg.norm(Error)/np.linalg.norm(Solution_Eval)
		[DFT_Error,Freqs]=[np.fft.fftshift(np.fft.fft(Error)),np.fft.fftshift(np.fft.fftfreq(len(Error),d=Dx))]
		Dictionary={'Time': Elapsed,'History': History,'Relative_L2_Error': Relative_L2_Error,'DFT_Error': DFT_Error,'DFT_Freqs': Freqs,'Solution': String_Multi,'Network_Eval': Network_Eval,'Solution_Eval': Solution_Eval.'Hidden_Layers': Hidden_Layers,'Neurons_Per_Layer': Neurons_Per_Layer}
		Name='Models_'+SigmaName+'/'+'Test_'+str(Test)+'/'+'Model_Multi_'+str(NR)+'NR_'+str(i+1)
		File=open(Name,'wb')
		dill.dump(Dictionary,File)
		File.close()
		Solver.Print_Cost()
		print('Model Saved Successfully')
