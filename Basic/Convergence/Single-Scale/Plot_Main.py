#! /usr/bin/python3

from Basic.PINN_Resolutors import *


""" Main To Visualize Error Plots For Basic PINN Single-Scale Convergence Analysis """


Criterion={'Average': jnp.mean,'Best': jnp.min}
def main():
	while (True):
		Continue=int(input('Choose:\n0: Leave\n1: Visualize\n'))
		if (Continue):
			T=str(input('Test: '))
			Act=str(input('Activation: '))
			HL=str(input('Hidden Layers: '))
			NPL=str(input('Neurons Per Layer: '))
			NumbRes=eval(input('Number Residuals: '))
			I=int(input('Instances: '))
			Latex=eval(input('Latex: '))
			Relative_L2_Error=np.zeros((I,len(NumbRes)))
			for i in range(I):
				for j in range(len(NumbRes)):
					Name='Test_'+T+'/'+Act+'_Model_'+str(HL)+'HL_'+str(NPL)+'NPL_'+str(NumbRes[j])+'NR_'+str(i+1)
					File=open(Name,'rb')
					Model=dill.load(File)
					Relative_L2_Error[i,j]=Model['Relative_L2_Error']
					File.close()
			for ModeName,Mode in Criterion.items():
				Plot_Vector=Mode(Relative_L2_Error,axis=0)
				plt.loglog(NumbRes,Plot_Vector,label='L2 Relative Error')
				plt.loglog(NumbRes,np.asarray(NumbRes,dtype=float)**(-1),label='Order 1')
				plt.loglog(NumbRes,np.asarray(NumbRes,dtype=float)**(-2),label='Order 2')
				plt.xticks(NumbRes,NumbRes)
				plt.legend()
				if (not Latex):
					plt.suptitle('Error '+ModeName+' Trend')
				plt.show()
		else:
			break


if __name__ == "__main__":
	main()
