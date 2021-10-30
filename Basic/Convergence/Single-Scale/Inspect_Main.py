#! /usr/bin/python3

from Basic.PINN_Resolutors import *


""" Inspect Main To Study Models Created For Basic PINN Single-Scale Convergence Analysis """


def main():
	while (True):
		Continue=int(input('Choose:\n0: Leave\n1: Inspect\n'))
		if (Continue):
			T=str(input('Test: '))
			Act=str(input('Activation: '))
			HL=str(input('Hidden Layers: '))
			NPL=str(input('Neurons Per Layer: '))
			NR=str(input('Number Residuals: '))
			I=int(input('Instances: '))
			for i in range(I):
				Name='Test_'+T+'/'+Act+'_Model_'+str(HL)+'HL_'+str(NPL)+'NPL_'+NR+'NR_'+str(i+1)
				File=open(Name,'rb')
				Model=dill.load(File)
				Fig1=plt.figure(1)
				Fig1.suptitle('Cost Evolution')
				plt.semilogy(Model['History'])
				Fig2=plt.figure(2)
				Fig2.suptitle('Network & Solution')
				plt.plot(Model['Points_Eval'],Model['Network_Eval'],label='Network')
				plt.plot(Model['Points_Eval'],Model['Solution_Eval'],label=Model['Solution'])
				plt.legend()
				plt.show()
				print('Attempt #'+str(i+1))
				print('Relative L2 Error: ', Model['Relative_L2_Error'])
				print('Total Learning Time: ', Model['Time'])
				File.close()
		else:
			break


if __name__ == "__main__":
	main()
