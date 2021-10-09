#! /usr/bin/python3

from Adaptive.PINN_Resolutors import *

def F(X):
	return (-jnp.pi**2)*(jnp.sin(10*jnp.pi*X))
def S(X):
	return jnp.sin(jnp.pi*X)
def G(X):
	return jnp.zeros_like(X)

RAR_Options={'Step': 50,'PoolSize': 1000,'MaxAdd': 100,'MaxResiduals': 1000}
NAE_Options={'Initial_Neurons': 30,'Relaxation': 0.5*np.ones(6),'DiffTol': 1e-6,'Neuron_Increase_Factor': 1.2,'Max_Hidden_Layers': 5}
AAF_Options={'Step': np.nan,'Learning_Rate': 1e-5,'Scaling_Increase_Factor': 2.5}
SAM_Options={'Step': 1000,'Learning_Rate': 5e-2}
AdaptiveFeatures={'RAR': RAR_Options,'NAE': NAE_Options,'AAF': AAF_Options,'SAM': SAM_Options}
Number_Residuals=20
ADAM_Batch_Fraction=0.5
ADAM_Steps=0
SecondOrder_MaxSteps=1000
Domain=np.array([[-1.0,1.0]])
Points=np.linspace(-1.0,1.0,20001)
Dx=Points[1]-Points[0]
Number_Boundary_Points=[[1,1]]
Boundary_Labels=2*['Dirichlet']
Sigma=jnp.tanh
TestPoints={'Internal': 10000,'Boundary': Number_Boundary_Points}
Solver=Resolutor_ADAM_BFGS_Adaptive[Poisson_Scalar_Adaptive](1,Sigma,AdaptiveFeatures,Domain,Number_Residuals,Number_Boundary_Points,Boundary_Labels,F,G,G)
A,B,C,D,E=Solver.Learn(ADAM_Steps,ADAM_Batch_Fraction,SecondOrder_MaxSteps,TestPoints)
Globals(Set={'Rows': D,'Cum': E})
Solver.Plot_1D(np.linspace(-1,1,20001),Name='End')
