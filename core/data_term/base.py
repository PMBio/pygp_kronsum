import numpy as np
import scipy as sp

class DataTerm(object):
	def __init__(self):
		pass

	@property
	def shape(self):
		raise NotImplementedError

	def value(self):
		raise NotImplementedError

	def raw(self):
		raise NotImplementedError

	def grad_theta(self, theta, i):
		raise NotImplementedError