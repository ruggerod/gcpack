import pytest
from gcpack.lin_regr import lregr
import numpy as np

def test_lregr():
	# Choose the "true" parameters.
	m_true = -0.9594
	b_true = 4.294

	# Generate some synthetic data from the model.
	N = 50
	x = np.sort(10*np.random.rand(N))
	yerr = 0.1 + 0.5*np.random.rand(N)
	y = m_true*x + b_true
	y += yerr * np.random.randn(N)

	# 1. does the fit approximately find the true values?
	th, therr = lregr(x, y, yerr)
	assert (abs(b_true - th[0]) / b_true < 0.05)
	assert (abs(m_true - th[1]) / m_true < 0.05)

	# 2. print info
	th, therr = lregr(x, y, yerr, verbose=True)