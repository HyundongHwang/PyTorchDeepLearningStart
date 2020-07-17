import myutil as mu
import numpy as np

t = np.array([0., 1., 2., 3., 4., 5., 6.])
mu.log("t", t)
mu.log("t.ndim", t.ndim)
mu.log("t.shape", t.shape)
mu.log("t[-1]", t[-1])
mu.log("t[2:5]", t[2:5])
mu.log("t[4:-1]", t[4:-1])
mu.log("t[:2]", t[:2])
mu.log("t[3:]", t[3:])

t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
mu.log("t", t)
mu.log("t.ndim", t.ndim)
mu.log("t.shape", t.shape)