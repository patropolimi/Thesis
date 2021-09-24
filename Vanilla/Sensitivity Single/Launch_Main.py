#! /usr/bin/python3

from PINN_Resolutors import *


""" Main Script [To Be Launched Once For Each Test] [To Be Tuned By Hand For Every Launch] -> PINN Sensitivity Analysis

	{In Each Test -> Change Only F_Low, F_Medium, F_High & Relative Solution Strings}

	Problem: Scalar 1D Poisson With Homogeneous Dirichlet Conditions In Domain [-1,1] [20 Residual Points]

	Launch To Create PINNs To Approximate:
	- Low Frequency Solution (F_Low: Corresponding Source Term)
	- Medium Frequency Solution (F_Medium: Corresponding Source Term)
	- High Frequency Solution (F_High: Corresponding Source Term)

	Trial Activation Functions:
	- ReLU (Rectified Linear Unit)
	- Tanh (Hyperbolic Tangent)
	- Sigmoid (Classic Sigmoid)

	Trial Architectures Parameters:
	- Hidden_Layers -> [1,2,3]
	- Neurons_Per_Layer -> [15,30,50]

	Three Models Trained With ADAM-BFGS (10000 ADAM Steps, 1000 Maximum BFGS Steps [Both Full-Batched & Default Settings]) For Each (Frequency-Architecture-Activation) Combination
	Each Model -> Saved In Proper Folder As Dictionary Containing:
	- Weights
	- Final Relative L2 Error
	- Elapsed Learning Time
	- History Cost Function
	- Discrete Fourier Transform Vector (Along With Frequencies)
	- Solution String """

ADAM_Steps=10000
ADAM_Batch=20
BFGS_MaxSteps=1000
NTests=3

def F_Low(X):
	return ((-jnp.pi**2)*jnp.sin(jnp.pi*X))
def Sol_Low(X):
	return (jnp.sin(jnp.pi*X))
String_Low='sin(pi*x)'

def F_Medium(X):
	return (-4*jnp.pi**2)*jnp.sin(2*jnp.pi*X)
def Sol_Medium(X):
	return (jnp.sin(2*jnp.pi*X))
String_Medium='sin(2*pi*x)'

def F_High(X):
	return (-25*jnp.pi**2)*jnp.sin(5*jnp.pi*X)
def Sol_High(X):
	return (jnp.sin(5*jnp.pi*X))
String_High='sin(5*pi*x)'

def G(X):
	return jnp.zeros_like(X)

Domain=np.array([[-1.0,1.0]])
Points=np.linspace(-1.0,1.0,2001)
dx=Points[1]-Points[0]
Number_Residuals=20
Number_Boundary_Points=[[1,1]]
Boundary_Labels=2*['Dirichlet']

Hidden_Layers=[1,2,3]
Neurons_Per_Layer=[15,30,50]

Activations={'Tanh': jnp.tanh,'Sigmoid': jax.nn.sigmoid,'ReLU': jax.nn.relu}
Sources={'Low': F_Low,'Medium': F_Medium,'High': F_High}
Solutions={'Low': Sol_Low,'Medium': Sol_Medium,'High': Sol_High}
Strings={'Low': String_Low,'Medium': String_Medium,'High': String_High}

def main():

	Test=int(input('Test Number: '))

	for SigmaName,Sigma in Activations.items():
		for HL in Hidden_Layers:
			for NPL in Neurons_Per_Layer:
				for Frequency,SRC in Sources.items():
					for i in range(NTests):
						Solver=Resolutor_ADAM_BFGS[Poisson_Scalar_Basic](1,HL,NPL,Sigma,Domain,Number_Residuals,Number_Boundary_Points,Boundary_Labels,SRC,G,G)
						Start=time.perf_counter()
						History=Solver.Learn(ADAM_Steps,ADAM_Batch,BFGS_MaxSteps)
						End=time.perf_counter()
						Elapsed=End-Start
						W=Solver.Weights
						Network_Eval=Solver.Network_Multiple(Points[None,:],W)[0,:]
						Solution=Solutions[Frequency]
						Solution_Eval=Solution(Points)
						Error=Solution_Eval-Network_Eval
						Relative_L2_Error=np.linalg.norm(Error)/np.linalg.norm(Solution_Eval)
						[DFT_Error,Freqs]=[np.fft.fft(Error),np.fft.fftfreq(len(Error),d=dx)]
						[DFT_Error,Freqs]=[np.fft.fftshift(DFT_Error),np.fft.fftshift(Freqs)]
						Dictionary={'W': W,'Time': Elapsed,'History': History,'Relative_L2_Error': Relative_L2_Error,'DFT_Error': DFT_Error,'Freqs_DFT': Freqs,'Solution': Strings[Frequency]}
						Name='Models_'+SigmaName+'/'+'Test_'+str(Test)+'/'+Frequency+'_'+str(HL)+'HL_'+str(NPL)+'NPL_'+str(i+1)
						File=open(Name,'wb')
						dill.dump(Dictionary,File)
						File.close()
						Solver.Print_Cost()
						print("Model Saved Successfully")


if __name__ == "__main__":
	main()
