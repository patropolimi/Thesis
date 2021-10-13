#! /usr/bin/python3

from Basic.PINN_Resolutors import *


""" Inspect Main To Study Models Created For Single-Scale Convergence Analysis Of Basic PINN

	Problem: Scalar 1D Poisson With Homogeneous Boundary Conditions In Domain [-1,1] """


Points=np.linspace(-1,1,20001)


def main():
	while (True):
		Continue=int(input('Choose:\n0: Leave\n1: Inspect\n'))
		if (Continue):
			Act=str(input('Activation: '))
			T=str(input('Test: '))
			NR=str(input('Number Residuals: '))
			I=str(input('Instance: '))
			Name='Models_'+Act+'/'+'Test_'+T+'/'+'Model_'+NR+'NR_'+I
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
			plt.plot(Model['DFT_Freqs'],np.abs(Model['DFT_Error']))
			plt.show()
			print('Relative L2 Error: ', Model['Relative_L2_Error'])
			print('Total Learning Time: ', Model['Time'])
			print('Solution: ', Model['Solution'])
			print('Hidden Layers: ', Model['Hidden_Layers'])
			print('Neurons Per Layer: ', Model['Neurons_Per_Layer'])
			File.close()
		else:
			break


if __name__ == "__main__":
	main()
