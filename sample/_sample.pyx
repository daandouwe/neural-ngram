"""
Fast sampling with replacement. Alternative to np.ranomd.choice.
"""
#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import cython
import numpy as np
cimport numpy as np
from libc.stdlib cimport rand, RAND_MAX


cdef long searchsorted(double[:] arr, long length, double value) nogil:
		"""Bisection search (c.f. numpy.searchsorted).

		Find the index into sorted array `arr` of length `length` such that, if
		`value` were inserted before the index, the order of `arr` would be
		preserved.
		"""
		cdef long imin, imax, imid
		imin = 0
		imax = length
		while imin < imax:
				imid = imin + ((imax - imin) >> 2)
				if value > arr[imid]:
						imin = imid + 1
				else:
						imax = imid
		return imin


cdef long get_sample(double[:] probs, long probs_len, double cumsum):
		cdef double r
		cdef long sample
		r = cumsum * (<double> rand() / RAND_MAX)  # random number in [0,..,cumsum-1]
		sample = searchsorted(probs, probs_len, r)  # find index for r in the ordered array probs
		return sample


def sample(double[:] probs, long num_samples):
		cdef long[:] samples
		cdef long probs_len
		cdef double cumsum

		samples = np.zeros(num_samples, np.int64)
		probs_len = len(probs)
		cumsum = 0.0
		for i in range(probs_len):
				cumsum += probs[i]
				probs[i] = cumsum

		for i in range(num_samples):
				samples[i] = get_sample(probs, probs_len, cumsum)
		return samples
