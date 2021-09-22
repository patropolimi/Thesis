#! /usr/bin/python3

from PINN_Resolutors import *


""" Main Script [To Be Launched Once For Each Test] -> PINN Sensitivity Analysis

	{In Each Test -> Change Only F_Low, F_Medium, F_High}

	Problem: Scalar 1D Poisson With Homogeneous Dirichlet Conditions [200 Residual Points]

	Launch To Create PINNs To Approximate (In Domain [-1,1]):
	- Low Frequency Solution (F_Low: Corresponding Source Term)
	- Medium Frequency Solution (F_Medium: Corresponding Source Term)
	- High Frequency Solution (F_High: Corresponding Source Term)

	Trial Activation Functions:
	- ReLU (Rectified Linear Unit)
	- Tanh (Hyperbolic Tangent)
	- Sigmoid (Classic Sigmoid)

	Trial Architectures Parameters:
	- Hidden_Layers -> [1,2,3,4,5]
	- Neurons_Per_Layer -> [10,25,50,100]

	Three Models Trained With ADAM-BFGS (Default Settings) For Each (Frequency-Architecture-Activation) Combination -> Saved In Proper Folder """


def F_Low(x):
	return (-jnp.pi**2)*jnp.sin(jnp.pi*x)

def F_Medium(x):
	return (-9*jnp.pi**2)*jnp.sin(3*jnp.pi*x)

def F_High(x):
	return (-100*jnp.pi**2)*jnp.sin(10*jnp.pi*x)

def G(x):
	return jnp.zeros_like(x)

Domain=np.array([[-1.0,1.0]])
Number_Residuals=5
Number_Boundary_Points=[[1,1]]
Boundary_Labels=['Dirichlet','Dirichlet']

Activations={'ReLU': jax.nn.relu,'Tanh': jnp.tanh,'Sigmoid': jax.nn.sigmoid}
Hidden_Layers=[1,2,3]
Neurons_Per_Layer=[10,25,50,100]

ADAM_Steps=[100,500,1000,2500]
ADAM_Batch=5
BFGS_MaxSteps=1000


def main():

	Test=int(input('Test Number: '))
	Sources={'Low': F_Low,'Medium': F_Medium,'High': F_High}

	for SigmaName,Sigma in Activations.items():
		for HL in Hidden_Layers:
			for NPL in Neurons_Per_Layer:
				for Frequency,SRC in Sources.items():
					for i in range(3):
						Solver=Resolutor_ADAM_BFGS[Poisson_Scalar_Basic](1,HL,NPL,Sigma,Domain,Number_Residuals,Number_Boundary_Points,Boundary_Labels,SRC,G,G)
						Solver.Learn(ADAM_Steps,ADAM_Batch,BFGS_MaxSteps)
						W=Solver.Weights
						Name='Models_'+str(SigmaName)+'/'+'Test_'+str(Test)+'/'+str(Frequency)+'_'+str(HL)+'HL_'+str(NPL)+'NPL_'+str(i+1)
						File=open(Name,'wb')
						dill.dump(W,File)
						File.close()
						print("Model Saved Successfully")


if __name__ == "__main__":
	main()
