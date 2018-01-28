# -*- coding: utf-8 -*-
"""
@Time    : 2018/1/3 18:51
@Author  : Elvis
"""
"""
 sae.py
  
"""
from scipy import linalg
import numpy as np

alpha = 0.5
x = np.random.rand(5, 5)
s = np.diag([1, 2, 1, 1, 2])
a = s * s.T
b = alpha * x * x.T
q = (1 + alpha) * s * x.T

w = linalg.solve_sylvester(a, b, q)
a.shape
b.shape
w.shape
np.allclose(a.dot(w) + w.dot(b), q)

c = a.dot(w)
c.shape