#! /usr/bin/python3

from PINN_Resolutors import *


""" Organizer Script [To Be Launched Once For Each Test] -> PINN Sensitivity Analysis

	Problem: Scalar 1D Poisson With Homogeneous Dirichlet Conditions In Domain [-1,1]

	This Script Gathers Models Results -> Creation Of Files With:
	- Relative L2 Error Average Table (Array) For Every Number Of Residuals
	- Relative L2 Error Best Table (Array) For Every Number Of Residuals

	Given Test Number -> To Be Tuned Below For Each Test
	For Each Combination Of (Frequency-Activation) -> Creation Of File With Tables Described Above """


Test=1
NAttempts=3
Number_Residuals=[25,100]
Hidden_Layers=[1,2,3]
Neurons_Per_Layer=[15,30,50]
Activations={'Tanh': jnp.tanh,'Sigmoid': jax.nn.sigmoid,'ReLU': jax.nn.relu}
Table_Modes={'Average': jnp.mean,'Best': jnp.min}
Frequencies=['VeryLow','Low','Medium','High','VeryHigh']
Hidden_Layers_Options=len(Hidden_Layers)
Neurons_Per_Layer_Options=len(Neurons_Per_Layer)


for SigmaName,Sigma in Activations.items():
		Dictionary={}
		for NR in Number_Residuals:
			for ModeName,Mode in Table_Modes.items():
				Relative_L2_Error=np.zeros((Neurons_Per_Layer_Options,Hidden_Layers_Options))
				for i in range(Neurons_Per_Layer_Options):
					for j in range(Hidden_Layers_Options):
						Element=[]
						for NA in range(NAttempts):
							LookName='./Models_'+SigmaName+'/'+'Test_'+str(Test)+'/'+'Model'+'_'+str(Hidden_Layers[j])+'HL_'+str(Neurons_Per_Layer[i])+'NPL_'+str(NR)+'NR_'+str(NA+1)
							File=open(LookName,'rb')
							D=dill.load(File)
							Element+=[D['Relative_L2_Error']]
							File.close()
						Relative_L2_Error[i,j]=Table_Modes[ModeName](jnp.asarray(Element))
				Dictionary[ModeName+'_'+str(NR)+'NR']=Relative_L2_Error
		SaveName='./Models_'+SigmaName+'/'+'Test_'+str(Test)+'/'+'Tables'
		File=open(SaveName,'wb')
		dill.dump(Dictionary,File)
		File.close()
		print("Results Saved Successfully")
