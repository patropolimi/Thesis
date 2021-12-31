README

===== INSTALLATION =====

REQUIRED:
- python (version 3)
- Clone this repository on your local machine

PACKAGES:
Run the following commands from terminal:
- pip install numpy
- pip install jaxlib
- pip install jax
- pip install matplotlib
- pip install dill
- pip install smt
- pip install type_templating

EXPORT PYTHONPATH:
Open the .bashrc file in your home directory and insert:
- export PYTHONPATH="${PYTHONPATH}:/your/path/to/Thesis/Library:/your/path/to/Thesis/Library/Basic:/your/path/to/Thesis/Library/Adaptive"
Quit the terminal and reopen it to make the modifications effective

===== DESCRIPTION =====

Riccardo Patroni's Thesis -> implementation of an elementary PINN library for solving PDEs over simple hyper-rectangular domains

CONTENT:
- Library -> folder containing the general PINN utilities along with the specific subfolders for the basic and adaptive plugins
- Basic -> folder containing all tests run with the basic PINN core & the relative scripts for their launch, organisation and visualisation
- Adaptive -> folder containing all the experiments run with the adaptive PINN core & the relative scripts for their launch, organisation and visualisation

FILES:
- Launch_Script.py -> run for prompting the training of the models for the relative test
- Inspect_Main.py -> run to visualise a category of trained models
- Organize_Main.py -> run to organise results for the sensitivity analysis related to the basic PINN models
- Result_Main.py -> run to visualise the results for the sensitivity analysis related to the basic PINN models
- Plot_Main.py -> run to visualise the error plots for the convergence analysis related to the basic PINN models
- All other files without a python extension represent an instance of a particular model: these are grouped in their relative test folder
