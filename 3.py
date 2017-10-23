# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 11:20:38 2016

@author: aries
"""

import numpy as np

a = np.array([1,2,3,4])
b = np.array([[1,2,3,4],[4,5,3,6],[5,6,7,8],[8,9,6,4]])
c = np.array([0,1,2,3])

for m, a in enumerate(b):
    a[3] = c[m]

    
