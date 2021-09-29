#! /usr/bin/python3

from PINN_Resolutors import *


""" Script To Create Plots For Single-Scale Convergence Analysis Of Basic PINN [To Be Launched Once For Each Test] [To Be Tuned For Each Launch]

	Problem: Scalar 1D Poisson With Homogeneous Boundary Conditions In Domain [-1,1] """


Test=1
NAttempts=3
Number_Residuals=[100,200,400,800,1600]
Relative_L2_Error=np.zeros((NAttempts,len(Number_Residuals)))
Criterion={'Average': jnp.mean,'Best': jnp.min}
Activations=['Tanh']

for SigmaName in Activations:
	for i in range(NAttempts):
		for j in range(len(Number_Residuals)):
			Name="Models_"+SigmaName+"/Test_"+str(Test)+"/Model_"+str(Number_Residuals[j])+"NR_"+str(i+1)
			File=open(Name,'rb')
			Model=dill.load(File)
			Relative_L2_Error[i,j]=Model['Relative_L2_Error']
			File.close()
	for ModeName,Mode in Criterion.items():
		Name="Models_"+SigmaName+"/Test_"+str(Test)+"/Models_"+ModeName+"_Plot"
		Plot_Vector=Mode(Relative_L2_Error,axis=0)
		Fig=plt.figure()
		plt.loglog(Number_Residuals,Plot_Vector)
		plt.xticks(Number_Residuals,Number_Residuals)
		plt.suptitle('Error '+ModeName+' Trend')
		plt.savefig(Name+'.eps')
		plt.close()
