#! /usr/bin/python3

from PINN_Problems_Adaptive import *


P=TemplateParameter('P')


class Resolutor_ADAM_BFGS_Adaptive(P,metaclass=Template[P]):
