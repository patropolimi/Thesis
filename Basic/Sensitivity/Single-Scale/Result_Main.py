#! /usr/bin/python3

from PINN_Resolutors import *


""" Main To Show Error Result Tables -> Basic PINN Sensitivity Analysis (Single-Scale)

	Problem: Scalar 1D Poisson With Homogeneous Dirichlet Conditions In Domain [-1,1] """


def main():
	print("Row Indexes -> Represent Different Neurons Per Layer Options (Increasing Order)")
	print("Column Indexes -> Represent Different Hidden Layers Options (Increasing Order)")
	while (True):
		Continue=int(input("\nChoose:\n0: Leave\n1: Show\n"))
		if (Continue):
			Act=str(input("Activation: "))
			T=str(input("Test: "))
			F=str(input("Frequency: "))
			Name="./Models_"+Act+"/Test_"+T+"/"+'Models_'+F+"_Tables"
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
