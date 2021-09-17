#! /usr/bin/python3

n_e=2000
nu=0.001
ro=0.9
bs=50
d=1e-8
Layers_1=[1,100,1]
Layers_1_1=[1,10000,1]
Layers_2=[1,100,1000,1]
Layers_3=[1,200,500,1000,1]
W1=Glorot(Layers_1,0)
W1_1=Glorot(Layers_1_1,0)
W2=Glorot(Layers_2,0)
W3=Glorot(Layers_3,0)
X=np.array(np.linspace(-1,1,20001))
X=X[None,:]
Y1=f1(X)
Y2=f2(X)
Y3=f3(X)
J=MSE
J=jax.jit(J)
Grad_J=jax.grad(J)
Grad_J=jax.jit(Grad_J)
Sigma=Sigmoid
Sigma=jax.jit(Sigma)

RMSPROP(X,Y3,W3,n_e,nu,bs,d,ro)
plt.plot(X[0,:],Y3[0,:],color='red')
plt.plot(X[0,:],DNN(W3,X)[0,:],color='blue')
plt.show()