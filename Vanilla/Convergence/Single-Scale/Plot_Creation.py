#! /usr/bin/python3

from PINN_Resolutors import *


""" Main Script To Create Plots For Convergence Analysis Of Vanilla PINN

	Problem: Scalar 1D Poisson With Homogeneous Boundary Conditions In Domain [-1,1] """


Test=1
Attempts=3
Residual_Number=[10,20,40,80,160,320]
Relative_L2_Error=np.zeros((Attempts,len(Residual_Number)))
Criterion={'Average': jnp.mean,'Best': jnp.min}
for i in range(Attempts):
	for j in range(len(Residual_Number)):
		Name="./Test_"+str(Test)+"/Model_"+str(Residual_Number[j])+"NR_"+str(i+1)
		File=open(Name,'rb')
		Model=dill.load(File)
		Relative_L2_Error[i,j]=Model['Relative_L2_Error']
		File.close()
for ModeName,Mode in Criterion.items():
	Name='./Test_'+str(Test)+'/'+ModeName+'_Plot'
	Plot_Vector=Mode(Relative_L2_Error,axis=0)
	Fig=plt.figure()
	plt.loglog(Residual_Number,Plot_Vector)
	plt.xticks(Residual_Number,Residual_Number)
	plt.suptitle('Error'+ModeName+'Trend')
	plt.savefig(Name+'.eps')
	plt.close()
