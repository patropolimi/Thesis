#! /usr/bin/python3

from Basic.PINN_Resolutors import *


""" Inspect Main To Study Models Created For Basic PINN Generic Sensitivity Analysis """


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
			D=int(input('Dimensions: '))
			Latex=eval(input('Latex: '))
			for i in range(I):
				Name='Test_'+T+'/'+Act+'_Model_'+HL+'HL_'+NPL+'NPL_'+NR+'NR_'+str(i+1)
				File=open(Name,'rb')
				Model=dill.load(File)
				Fig1=plt.figure(1)
				if (not Latex):
					Fig1.suptitle('Cost Evolution')
				plt.semilogy(Model['History'])
				Fig2=plt.figure(2)
				if (not Latex):
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
					SliceTime=eval(input('Slice-Time Plots: '))
					while (SliceTime):
						TimeStep=int(eval(input('Time Step: ')))
						AbscissaSlice=YG[:,TimeStep]
						NetworkSlice=(Model['Network_Eval'].reshape((NY,NX)))[:,TimeStep]
						ExactSlice=(Model['Solution_Eval'].reshape((NY,NX)))[:,TimeStep]
						plt.plot(AbscissaSlice,NetworkSlice,label='Network')
						plt.plot(AbscissaSlice,ExactSlice,label='Solution')
						plt.legend()
						plt.show()
						SliceTime=eval(input('Slice-Time Plots: '))
				print('Attempt #'+str(i+1))
				print('Relative L2 Error: ', Model['Relative_L2_Error'])
				print('Total Learning Time: ', Model['Time'])
				File.close()
		else:
			break


if __name__ == "__main__":
	main()
