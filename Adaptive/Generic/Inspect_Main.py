#! /usr/bin/python3

from Adaptive.PINN_Resolutors import *


""" Inspect Main To Study Models Created For Adaptive PINN Generic Experiments """


def main():
	while (True):
		Continue=int(input('Choose:\n0: Leave\n1: Inspect\n'))
		if (Continue):
			T=str(input('Test: '))
			I=int(input('Instances: '))
			D=int(input('Dimensions: '))
			for i in range(I):
				Name='Test_'+str(T)+'/'+'Model_'+str(i+1)
				File=open(Name,'rb')
				Model=dill.load(File)
				Fig1=plt.figure(1)
				Fig1.suptitle('Salient Cost Evolution')
				plt.semilogy(Model['Saliency_History'])
				Fig2=plt.figure(2)
				Fig2.suptitle('Network & Solution')
				if (D==1):
					plt.plot(Model['Points_Eval'],Model['Network_Eval'],label='Network')
					plt.plot(Model['Points_Eval'],Model['Solution_Eval'],label=Model['Solution'])
					plt.legend()
					plt.show()
				if (D==2):
					XG,YG=np.meshgrid(Model['Points_Abscissa'],Model['Points_Ordinate'])
					NX=Model['Points_Abscissa'].shape[0]
					NY=Model['Points_Ordinate'].shape[0]
					Axes=plt.axes(projection='3d')
					Axes.plot_surface(XG,YG,Model['Network_Eval'].reshape((NY,NX)),label='Network')
					Axes.plot_surface(XG,YG,Model['Solution_Eval'].reshape((NY,NX)),label=Model['Solution'])
					plt.show()
				print('Attempt #'+str(i+1))
				print('Relative L2 Error: ', Model['Relative_L2_Error'])
				print('Total Learning Time: ', Model['Time'])
				print('Structure: ', Model['Structure'])
				print('Final Number Residuals: ', Model['Final_Number_Residuals'])
				File.close()
		else:
			break


if __name__ == "__main__":
	main()
