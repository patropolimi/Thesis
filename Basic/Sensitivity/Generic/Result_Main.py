#! /usr/bin/python3

from Basic.PINN_Resolutors import *


""" Main To Show Error Result Tables For Basic PINN Generic Sensitivity Analysis """


def main():
	print('Row Indexes -> Represent Different Neurons Per Layer Options In Increasing Order')
	print('Column Indexes -> Represent Different Hidden Layers Options In Increasing Order')
	while (True):
		Continue=int(input('Choose:\n0: Leave\n1: Show\n'))
		if (Continue):
			T=str(input('Test: '))
			Act=str(input('Activation: '))
			Name='Test_'+T+'/'+Act+'_Models_Tables'
			File=open(Name,'rb')
			Model=dill.load(File)
			File.close()
			for TableName,Table in Model.items():
				print('\n'+TableName+'\n')
				print(Table)
		else:
			break


if __name__ == "__main__":
	main()
