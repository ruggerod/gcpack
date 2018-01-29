import numpy as np
import matplotlib.pyplot as plt

def lregr(x, y, yerr, verbose=False, plot=False, labels=None):
	"""
	Calculate parameters (and errors) of the linear regression relative to m 
	data points with n features.
	If verbose is True, print regression info.

	
	Arguments
		x:			array([m,n]), with n number of features
		y:			array([m,1]), dependent variable
		yerr:		array([m,1]), errors on the dependent variable

		verbose:	bool, print linear regression info. Default is False
		plot:		bool, generate a plot (only available for 1d fit). Default
					id False
		labels:		tuple of str, (xlabel, ylabel) represent the quantities
					on the axes of the plot 
	
	Return
		theta:		array([n+1,1]), best fit parameters
		therr:		array([n+1,1]), parameters errors from the linear fit
	"""
	A = np.vstack((np.ones_like(x), x)).T # design matrix
	C = np.diag(yerr * yerr) # y covariance matrix
	# calculate theta covariance matrix
	cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A))) 
	theta = np.dot(cov, np.dot(A.T, np.linalg.solve(C, y))) # best fit params
	therr = np.sqrt(np.diag(cov)) # best fit params errors
	
	if verbose:
		res = y - np.dot(A, theta)
		chi2 = np.dot(res.T, np.linalg.solve(C, res))
		ndim = len(theta)
		print "{:<30}".format("AIC ="), \
			chi2 + 2. * ndim + (2. * ndim * (ndim+1) ) / (len(y) - ndim - 1)
		print "{:<30}".format("best-fit chi-square ="), chi2

	if plot:
		if len(np.shape(x)) > 1:
			raise ValueError("Plot option only available for 1D linear fit.")
		plt.errorbar(x, y, yerr=yerr, fmt='o')
		plt.plot(x,  x * theta[1] + theta[0], '-')
		if labels is None:
			plt.xlabel('x')
			plt.ylabel('y')
		else:
			plt.xlabel(labels[0])
			plt.ylabel(labels[1])
		plt.show()

	return theta, therr


