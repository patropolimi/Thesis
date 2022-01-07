#! /usr/bin/python3

from Basic.PINN_Resolutors import *


""" Inspect Main To Study Models Created For Basic PINN Single-Scale Sensitivity Analysis """


def main():
	while (True):
		Continue=int(input('Choose:\n0: Leave\n1: Inspect\n'))
		if (Continue):
			T=str(input('Test: '))
			Act=str(input('Activation: '))
			F=str(input('Frequency: '))
			HL=str(input('Hidden Layers: '))
			NPL=str(input('Neurons Per Layer: '))
			NR=str(input('Number Residuals: '))
			I=int(input('Instances: '))
			Latex=eval(input('Latex: '))
			for i in range(I):
				Name='Test_'+T+'/'+Act+'_Model_'+F+'_'+HL+'HL_'+NPL+'NPL_'+NR+'NR_'+str(i+1)
				File=open(Name,'rb')
				Model=dill.load(File)
				Fig1=plt.figure(1)
				if (not Latex):
					Fig1.suptitle('Cost Evolution')
				plt.semilogy(Model['History'])
				Fig2=plt.figure(2)
				if (not Latex):
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
