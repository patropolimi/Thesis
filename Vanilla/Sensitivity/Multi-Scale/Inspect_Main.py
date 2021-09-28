#! /usr/bin/python3

from PINN_Resolutors import *


""" Main Script To Inspect Models Created For Sensitivity Analysis Of Vanilla PINN

	Problem: Scalar 1D Poisson With Homogeneous Boundary Conditions In Domain [-1,1] """


Points=np.linspace(-1,1,2001)


def main():
	while (True):
		Continue=int(input("Choose:\n0: Leave\n1: Inspect Model\n"))
		if (Continue):
			Act=str(input("Activation: "))
			T=str(input("Test Number: "))
			HL=str(input("Hidden Layers: "))
			NPL=str(input("Neurons Per Layer: "))
			NR=str(input("Number Residuals: "))
			I=str(input("Instance: "))
			Name="./Models_"+Act+"/Test_"+T+"/"+'Model'+"_"+HL+"HL_"+NPL+"NPL_"+NR+"NR_"+I
			File=open(Name,'rb')
			Model=dill.load(File)
			Fig1=plt.figure(1)
			Fig1.suptitle('Cost Evolution')
			plt.semilogy(Model['History'])
			plt.show()
			Fig2=plt.figure(2)
			Fig2.suptitle('Network & Solution')
			plt.plot(Points,Model['Network_Eval'])
			plt.plot(Points,Model['Solution_Eval'])
			plt.show()
			Fig3=plt.figure(3)
			Fig3.suptitle('Error Discrete Fourier Transform (Absolute Value)')
			plt.plot(Model['Freqs_DFT'], np.abs(Model['DFT_Error']))
			plt.show()
			print("Relative L2 Error: ", Model['Relative_L2_Error'])
			print("Total Learning Time: ", Model['Time'])
			print("Solution: ", Model['Solution'])
			print("Batch: ", Model['Batch_Size'])
			File.close()
		else:
			break


if __name__ == "__main__":
	main()
