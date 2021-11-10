#! /usr/bin/python3

from Basic.PINN_Resolutors import *


""" Organize Main To Write Summary Error Tables For Models Created For Basic PINN Generic Sensitivity Analysis """


Table_Modes={'Average': jnp.mean,'Best': jnp.min}
def main():
	T=str(input('Test: '))
	Act=str(input('Activation: '))
	HiddLays=eval(input('Hidden Layers: '))
	NeursPerLay=eval(input('Neurons Per Layer: '))
	NumbRes=eval(input('Number Residuals: '))
	NAtts=int(input('Instances: '))
	Dictionary={}
	for NR in NumbRes:
		for ModeName,Mode in Table_Modes.items():
			Relative_L2_Error=np.zeros((len(NeursPerLay),len(HiddLays)))
			for i,NPL in enumerate(NeursPerLay):
				for j,HL in enumerate(HiddLays):
					Element=[]
					for NA in range(NAtts):
						LookName='Test_'+str(T)+'/'+Act+'_Model_'+str(HL)+'HL_'+str(NPL)+'NPL_'+str(NR)+'NR_'+str(NA+1)
						File=open(LookName,'rb')
						D=dill.load(File)
						Element+=[D['Relative_L2_Error']]
						File.close()
					Relative_L2_Error[i,j]=Mode(jnp.asarray(Element))
			Dictionary[ModeName+'_'+str(NR)+'NR']=Relative_L2_Error
	SaveName='Test_'+str(T)+'/'+Act+'_Models_Tables'
	File=open(SaveName,'wb')
	dill.dump(Dictionary,File)
	File.close()
	print('Results Saved Successfully')


if __name__ == "__main__":
	main()
