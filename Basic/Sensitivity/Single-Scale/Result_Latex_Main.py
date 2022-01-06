#! /usr/bin/python3

from Basic.PINN_Resolutors import *


""" Main To Show Latex Code Translation Of Error Result Tables For Basic PINN Single-Scale Sensitivity Analysis """


def main():
	while (True):
		Continue=int(input('Choose:\n0: Leave\n1: Show\n'))
		if (Continue):
			T=str(input('Test: '))
			Act=str(input('Activation: '))
			F=str(input('Frequency: '))
			Name='Test_'+T+'/'+Act+'_Models_'+F+'_Tables'
			File=open(Name,'rb')
			Model=dill.load(File)
			File.close()
			for TableName,Table in Model.items():
				Latex_Table=pd.DataFrame(Table)
				print('\n'+TableName+'\n')
				print(Latex_Table.to_latex())
		else:
			break


if __name__ == "__main__":
	main()
