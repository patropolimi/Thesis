#! /usr/bin/python3

from PINN_Resolutors import *


""" Script To Launch Models Singularly [To Be Tuned By Hand For Every Launch] -> PINN Sensitivity Analysis

	Problem: Scalar 1D Poisson With Homogeneous Dirichlet Conditions In Domain [-1,1] """


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

def G(X):
	return jnp.zeros_like(X)

Number_Residuals=25
ADAM_Batch=25
ADAM_Steps=10000
BFGS_MaxSteps=10000
Domain=np.array([[-1.0,1.0]])
Points=np.linspace(-1.0,1.0,2001)
Dx=Points[1]-Points[0]
Number_Boundary_Points=[[1,1]]
Boundary_Labels=2*['Dirichlet']
Activations={'Tanh': jnp.tanh,'Sigmoid': jax.nn.sigmoid,'ReLU': jax.nn.relu}
Sources={'VeryLow': F_VeryLow,'Low': F_Low,'Medium': F_Medium,'High': F_High,'VeryHigh': F_VeryHigh}
Solutions={'VeryLow': Sol_VeryLow,'Low': Sol_Low,'Medium': Sol_Medium,'High': Sol_High,'VeryHigh': Sol_VeryHigh}
Strings={'VeryLow': String_VeryLow,'Low': String_Low,'Medium': String_Medium,'High': String_High,'VeryHigh': String_VeryHigh}
Test=1
Attempt=1
Hidden_Layers=1
Neurons_Per_Layer=100
SigmaName='Tanh'
Frequency='High'



Solver=Resolutor_ADAM_BFGS[Poisson_Scalar_Basic](1,Hidden_Layers,Neurons_Per_Layer,Activations[SigmaName],Domain,Number_Residuals,Number_Boundary_Points,Boundary_Labels,Sources[Frequency],G,G)
Start=time.perf_counter()
History=Solver.Learn(ADAM_Steps,ADAM_Batch,BFGS_MaxSteps)
End=time.perf_counter()
Elapsed=End-Start
W=copy.deepcopy(Solver.Weights)
Network_Eval=Solver.Network_Multiple(Points[None,:],W)[0,:]
Solution=Solutions[Frequency]
Solution_Eval=Solution(Points)
Error=Solution_Eval-Network_Eval
Relative_L2_Error=np.linalg.norm(Error)/np.linalg.norm(Solution_Eval)
[DFT_Error,Freqs]=[np.fft.fft(Error),np.fft.fftfreq(len(Error),d=Dx)]
[DFT_Error,Freqs]=[np.fft.fftshift(DFT_Error),np.fft.fftshift(Freqs)]
Dictionary={'W': W,'Time': Elapsed,'History': History,'Relative_L2_Error': Relative_L2_Error,'DFT_Error': DFT_Error,'Freqs_DFT': Freqs,
			'Solution': Strings[Frequency],'Network_Eval': Network_Eval,'Solution_Eval': Solution_Eval,'Batch_Size': ADAM_Batch}
Name='Models_'+SigmaName+'/'+'Test_'+str(Test)+'/'+Frequency+'_'+str(Hidden_Layers)+'HL_'+str(Neurons_Per_Layer)+'NPL_'+str(Number_Residuals)+'NR_'+str(Attempt)
File=open(Name,'wb')
dill.dump(Dictionary,File)
File.close()
Solver.Print_Cost()
print("Model Saved Successfully")
