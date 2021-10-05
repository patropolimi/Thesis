#! /usr/bin/python3

from Adaptive.PINN_Problems import *


P=TemplateParameter('P')


class Resolutor_ADAM_BFGS_Adaptive(P,metaclass=Template[P]):
