===== INSTALLATION =====

REQUIRED:
- python 3
- Clone this repository on your local machine

PACKAGES:
Run the following commands from terminal:
- pip install numpy
- pip install jaxlib
- pip install jax
- pip install smt
- pip install type_templating
- pip install matplotlib

EXPORT PYTHONPATH:
Open the .bashrc file in your home directory and insert:
- export PYTHONPATH="${PYTHONPATH}:/your/path/to/this/repository/folder/Library"
Quit the terminal and reopen it to make the modifications effective

===== DESCRIPTION =====

Riccardo Patroni's Thesis Project.
The aim is to provide a simple deep learning machinery that implements adaptive PINNs in hyper-rectangular domains.
With this in mind, the following steps are performed:
- Reproduce the results found in literature for the basic PINN architecture over a set of test cases.
- Provide a variety of adaptive tools for PINNs that are to be tested.
- Discuss the results obtained with this approach.

Adaptive Techniques:
- Pruning & Growing.
- Residual Adaptive Refinement.
- Soft Attention Mechanism.
- Adaptive Activation Function.

===== CONTENT ======

- Library -> Contains all the utilities and class implementations needed.
	- PINN_Utilities.py -> Utilities.
	- PINN_Grounds.py -> Implementation of the base classes PINN_Basic & Geometry_Basic.
	- PINN_Wrappers.py -> Implementation of the base class Problem_Scalar_Basic for scalar-output problems, which derives from PINN_Basic & Geometry_Basic.
	- PINN_Problems.py -> Implementation of the currently available test problems classes, deriving from the properly-correspondent wrapper.
	- PINN_Resolutors.py -> Implementation of the currently available solvers for PINNs, each deriving from the underlying problem class.
