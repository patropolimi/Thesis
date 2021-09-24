#! /usr/bin/python3

from PINN_Resolutors import *


""" Main Script To Inspect Models Created For Sensitivity Analysis Of Vanilla PINN

	Problem: Scalar 1D Poisson With Homogeneous Boundary Conditions In Domain [-1,1] """

Activations={'Tanh': jnp.tanh,'Sigmoid': jax.nn.sigmoid,'ReLU': jax.nn.relu}
Points=np.linspace(-1,1,2001)

# Fictitious Source Term
def F(X):
	return jnp.zeros_like(X)
# Fictitious Boundary Term
def G(X):
	return jnp.zeros_like(X)
# Fictitious Solver
Solver=Resolutor_ADAM_BFGS[Poisson_Scalar_Basic](1,1,1,None,np.array([[-1.0,1.0]]),1,[[1,1]],2*['Dirichlet'],F,G,G)

def main():
	print("\nTest 1 Made On Slower Machine With 20 Residuals\nTest 2 Made On Faster Machine With 20 Residuals\nSpecial Tests Were Made On Faster Machine With Different Number Of Residuals")
	print("Test 1& Test 2 Models Were Trained By ADAM-BFGS -> 10000 ADAM Steps Full-Batch & Maximum 1000 BFGS Steps\n")
	print("WARNING: Some Optimization Settings Were Slightly Changed During The Gathering Of All These Models & Results\n")
	while (True):
		Continue=int(input("Choose:\n0: Leave\n1: Classic Model\n2: Special Model\n"))
		if (Continue):
			if (Continue==1):
				Act=str(input("Activation: "))
				T=str(input("Test Number: "))
				F=str(input("Frequency: "))
				HL=str(input("Hidden Layers: "))
				NPL=str(input("Neurons Per Layer: "))
				I=str(input("Instance: "))
				Solver.Activation=Activations[Act]
				Name="./Models_"+Act+"/Test_"+T+"/"+F+"_"+HL+"HL_"+NPL+"NPL_"+I
				File=open(Name,'rb')
				Model=dill.load(File)
				Fig1=plt.figure(1)
				Fig1.suptitle('Cost Evolution (Sampling Step: 10)')
				plt.semilogy(Model['History'])
				plt.show()
				Fig2=plt.figure(2)
				Fig2.suptitle('Network Plot')
				Solver.Plot_1D(Points,Model['W'])
				Fig3=plt.figure(3)
				Fig3.suptitle('Error Discrete Fourier Transform (Absolute Value)')
				plt.plot(Model['Freqs_DFT'], np.abs(Model['DFT_Error']))
				plt.show()
				print("Relative L2 Error: ", Model['Relative_L2_Error'])
				print("Total Learning Time: ", Model['Time'])
				print("Solution: ", Model['Solution'])
				File.close()
			elif (Continue==2):
				ST=str(input("Special Model ID: "))
				Name="./Extra/Model_"+ST
				File=open(Name,'rb')
				Model=dill.load(File)
				Solver.Activation=Activations[Model['Activation']]
				Fig1=plt.figure(1)
				Fig1.suptitle('Cost Evolution')
				plt.semilogy(Model['History'])
				plt.show()
				Fig2=plt.figure(2)
				Fig2.suptitle('Network Plot')
				Solver.Plot_1D(Points,Model['W'])
				Fig3=plt.figure(3)
				Fig3.suptitle('Error Discrete Fourier Transform (Absolute Value)')
				plt.plot(Model['Freqs_DFT'], np.abs(Model['DFT_Error']))
				plt.show()
				print("Number Residuals: ", Model['Number_Residuals'])
				print("ADAM Steps: ", Model['ADAM_Steps'])
				print("ADAM Batch:", Model['ADAM_Batch'])
				print("BFGS Max Steps:", Model['BFGS_MaxSteps'])
				print("Hidden Layers: ", Model['Hidden_Layers'])
				print("Neurons Per Layer: ", Model['Neurons_Per_Layer'])
				print("Relative L2 Error: ", Model['Relative_L2_Error'])
				print("Total Learning Time: ", Model['Time'])
				print("Solution: ", Model['Solution'])
				File.close()
		else:
			break


if __name__ == "__main__":
	main()
