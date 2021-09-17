#! /usr/bin/python3


class OBD_Tools:

	def __init__(self,PINN):

		self.PINN=PINN
		self.Activation=PINN.Activation
		self.Grad_Activation=jax.grad(self.Activation)

	def FW(self,X,W,A,M,N):
		L=len(W)
		Z=[]
		Y=X
		for l in range(L-1):
			Y=self.Activation(N[l]*A[l]*(((W[l][:,:-1]*M[l][:,:-1])@Y)+(W[l][:,-1:]*M[l][:,-1,:])))
		Z.append(Y)
		Z.append(self.Activation(N[L-1]*A[L-1]*(((W[L-1][:,:-1]*M[L-1][:,:-1])@Y)+(W[L-1][:,-1:]*M[L-1][:,-1,:]))))
		return Z
	FW=jax.jit(FW)

	def BW(self,W,Z):






def BackwardPass(Weights,Z,YTrue):      # Backward Pass (For Automatic Differentiation)
  # ErrorBP is a list containing:
  # - Partial derivative of the cost function with respect to the inputs of every hidden layer
  # - Partial derivative of the cost function with respect to the inputs of the output layer
  L=len(Weights)
  ParD_J__ParD_DNN_Out=Fun_ParD_J__ParD_DNN_Out(Z[-1],YTrue)
  ErrorBP=[StripeAppJacT(Z[-2],ParD_J__ParD_DNN_Out)]
  for l in range(L-1):
    ErrorBP.append(((Weights[L-1-l][:,:-1].T)@(ErrorBP[l]))*(CompoGrad(Grad_Sigma,Z[L-l-1])))
  list.reverse(ErrorBP)
  return ErrorBP
BackwardPass=jax.jit(BackwardPass)


def CompoGrad(Grad_f,X):
  # Grad_f: gradient of the scalar function f
  # X: bidimensional array
  # Returns the bidimensional array containing the element-wise application of Grad_f to X
  return jnp.asarray(jnp.reshape(jax.vmap(Grad_f)(jnp.ravel(X)),jnp.shape(X)))

def SingleAppJacT(x,v):     # Utility For StripeAppJacT
  Jac,VJP_Fun=jax.vjp(Softener,x)
  return jnp.transpose(VJP_Fun(v)[0])
SingleAppJacT=jax.jit(SingleAppJacT)

def StripeAppJacT(X,V):
  # Jac_f: jacobian of the vector-valued function f
  # X,V: bidimensional arrays
  # Returns the bidimensional array whose n-th column is the application of Jac_f(X[:,n]).T to the vector V[:,n]
  return jnp.asarray(jax.vmap(SingleAppJacT,in_axes=(1,1),out_axes=1)(X,V))

def GradientConcatenation(delta,z):     # Utility For GradientComputation
  return jnp.concatenate(((delta[:,None])@(z[None,:]),delta[:,None]),axis=1)
GradientConcatenation=jax.jit(GradientConcatenation)

def GradientComputation(Z,Delta):
  # Delta is the alias of ErrorBP used in BackwardPass
  # GradW contains the gradient of the cost function with respect to the model weights
  L=len(Z)-2
  GradW=[jnp.sum(jnp.asarray(jax.vmap(GradientConcatenation,in_axes=(1,1),out_axes=1)(Delta[0],Z[0])),axis=1)]
  GradW+=[jnp.sum(jnp.asarray(jax.vmap(GradientConcatenation,in_axes=(1,1),out_axes=1)(Delta[l],Sigma(Z[l]))),axis=1) for l in range(1,L)]
  return GradW
GradientComputation=jax.jit(GradientComputation)

def BuildFinalD2(ZL,DNN_Out,Y):
  # ZL: input of the output layer
  # DNN_Out: output of the network (after the application of the softener)
  # Y: expected output
  # Returns the array whose n-th component is the second derivative of the cost function with respect to the n-th input of the output layer
  Dim=(ZL.shape)[0]
  First_Term=(((Jac_Softener(ZL).T)@(Fun_ParD2_J__ParD2_DNN_Out(DNN_Out,Y))@(Jac_Softener(ZL))).diagonal(0))[:,None]
  Vector=Fun_ParD_J__ParD_DNN_Out(DNN_Out,Y)[:,None]
  return jnp.asarray(First_Term)
BuildFinalD2=jax.jit(BuildFinalD2)

def DotTranspose(x,y):      # Utility For SecondBackwardDiagPass
  return jnp.dot(x[:,None],jnp.transpose(y)[None,:])
DotTranspose=jax.jit(DotTranspose)

def StepBack(W,z,temp,delta):     # Utility For SecondBackwardDiagPass
  return (CompoGrad(Grad_Sigma,z[:,None])**2)*(((W[:,:-1].T)**2)@(temp[:,None]))+(CompoGrad(Grad2_Sigma,z[:,None]))*((W[:,:-1].T)@(delta[:,None]))
StepBack=jax.jit(StepBack)

def SecondBackwardDiagPass(Weights,Z,Y,Delta):
  # Returns the approximate estimation of the second derivative of the cost function with respect to the model weights, according to "Optimal Brain Damage"
  L=len(Weights)
  N=(Z[0].shape)[1]
  Diagonal_Hessian=[jnp.zeros(w.shape) for w in Weights]
  Temp=jnp.asarray(jax.vmap(BuildFinalD2,in_axes=(1,1,1),out_axes=1)(Z[-2],Z[-1],Y))[:,:,0]
  for l in range(L-1):
    Diagonal_Hessian[-1-l]=index_add(Diagonal_Hessian[-1-l],index[:,-1],jnp.mean(Temp,axis=1).ravel())
    Diagonal_Hessian[-1-l]=index_add(Diagonal_Hessian[-1-l],index[:,:-1],jnp.mean(jnp.asarray(jax.vmap(DotTranspose,in_axes=(1,1))(Temp,Sigma(Z[-3-l])**2)),axis=0))
    Temp=jnp.asarray(jax.vmap(StepBack,in_axes=(None,1,1,1),out_axes=1)(Weights[-1-l],Z[-3-l],Temp,Delta[-1-l]))[:,:,0]
  Diagonal_Hessian[-L]=index_add(Diagonal_Hessian[-L],index[:,-1],jnp.mean(Temp,axis=1).ravel())
  Diagonal_Hessian[-L]=index_add(Diagonal_Hessian[-L],index[:,:-1],jnp.mean(jnp.asarray(jax.vmap(DotTranspose,in_axes=(1,1))(Temp,(Z[-2-L])**2)),axis=0))
  return Diagonal_Hessian
SecondBackwardDiagPass=jax.jit(SecondBackwardDiagPass)

def Saliency(Weights,Diag_H):     # Saliency Computation
  return [(1/2)*(Diag_H[l])*(Weights[l]**2) for l in range(len(Weights))]
Saliency=jax.jit(Saliency)