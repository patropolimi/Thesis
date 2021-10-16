#! /usr/bin/python3

from Basic.PINN_Resolutors import *


""" Inspect Main To Study Models Created For Wave 1D """


NT=1001
NX=1001
PT=np.linspace(0.0,1.0,NT)
PX=np.linspace(0.0,1.0,NX)


def main():
	while (True):
		Continue=int(input('Choose:\n0: Leave\n1: Inspect\n'))
		if (Continue):
			Act=str(input('Activation: '))
			T=str(input('Test: '))
			HL=str(input('Hidden Layers: '))
			NPL=str(input('Neurons Per Layer: '))
			NR=str(input('Number Residuals: '))
			I=str(input('Instance: '))
			Name='Models_'+Act+'/'+'Test_'+T+'/'+'Model_'+HL+'HL_'+NPL+'NPL_'+NR+'NR_'+I
			File=open(Name,'rb')
			Model=dill.load(File)
			Fig1=plt.figure(1)
			Fig1.suptitle('Cost Evolution')
			plt.semilogy(Model['History'])
			plt.show()
			Fig2=plt.figure(2)
			Fig2.suptitle('Network & Solution')
			TG,XG=np.meshgrid(PT,PX)
			Axes=plt.axes(projection='3d')
			Axes.plot_surface(TG,XG,Model['Network_Eval'].reshape((NX,NT)))
			Axes.plot_surface(TG,XG,Model['Solution_Eval'].reshape((NX,NT)))
			plt.show()
			print('Relative L2 Error: ', Model['Relative_L2_Error'])
			print('Total Learning Time: ', Model['Time'])
			print('Solution: ', Model['Solution'])
			File.close()
		else:
			break


if __name__ == "__main__":
	main()
