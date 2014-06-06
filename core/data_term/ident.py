import numpy as np
import scipy as sp
from base import DataTerm

class IdentDT(DataTerm):
	def __init__(self, rawY):
		assert type(rawY) is np.ndarray
		assert len(rawY.shape) == 2

		self._rawY = rawY
		self._nrows = rawY.shape[0]
		self._ncols = rawY.shape[1]

	@property
	def shape(self):
		return (self._nrows, self._ncols)

	def value(self):
		return self._rawY

	def raw(self):
		return self._rawY

	def grad_theta(self, theta, i):
		assert theta is np.ndarray
		assert theta.size == 0
		assert i is None

		return np.full((self._nrows, self._ncols), None, dtype=object)