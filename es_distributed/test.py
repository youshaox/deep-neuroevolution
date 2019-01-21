#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Created by youshaox on 3/1/19
"""
function:

"""
import sys
import numpy as np
#解决 二进制str 转 unicode问题
# reload(sys)
# sys.setdefaultencoding('utf8')


k=3
distances = np.array([5,1,2,3,4])
print(distances)
top_k_indicies = (distances).argsort()[:k]
top_k = distances[top_k_indicies]
print(top_k)