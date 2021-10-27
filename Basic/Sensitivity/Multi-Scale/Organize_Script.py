#! /usr/bin/python3

from Basic.PINN_Resolutors import *


""" Organizer Script [To Be Launched Once For Each Test] [To Be Tuned For Each Launch] -> Basic PINN Sensitivity Analysis (Multi-Scale)

	Problem: Scalar 1D Poisson With Homogeneous Dirichlet Conditions In Domain [-1,1] """


Test=1
NAttempts=3
Number_Residuals=[250,2500]
Hidden_Layers=[1,2]
Neurons_Per_Layer=[40,80]
Activations=['Tanh']
Table_Modes={'Average': jnp.mean,'Best': jnp.min}


for SigmaName in Activations:
		Dictionary={}
		for NR in Number_Residuals:
			for ModeName,Mode in Table_Modes.items():
				Relative_L2_Error=np.zeros((len(Neurons_Per_Layer),len(Hidden_Layers)))
				for i,NPL in enumerate(Neurons_Per_Layer):
					for j,HL in enumerate(Hidden_Layers):
						Element=[]
						for NA in range(NAttempts):
							LookName='Models_'+SigmaName+'/'+'Test_'+str(Test)+'/'+'Model_Multi_'+str(HL)+'HL_'+str(NPL)+'NPL_'+str(NR)+'NR_'+str(NA+1)
							File=open(LookName,'rb')
							D=dill.load(File)
							Element+=[D['Relative_L2_Error']]
							File.close()
						Relative_L2_Error[i,j]=Mode(jnp.asarray(Element))
				Dictionary[ModeName+'_'+str(NR)+'NR']=Relative_L2_Error
		SaveName='Models_'+SigmaName+'/'+'Test_'+str(Test)+'/'+'Models_Multi_Tables'
		File=open(SaveName,'wb')
		dill.dump(Dictionary,File)
		File.close()
		print('Results Saved Successfully')
