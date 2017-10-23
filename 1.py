# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 15:56:39 2016

@author: H81
"""
"""

import tensorflow as tf
sess = tf.InteractiveSession()
x = tf.Variable([1.0, 2.0])
b = tf.constant([3.0, 3.0])
x.initializer.run()
sud = tf.sub(x, b)
print sud.eval()
"""

# 进入一个交互式Tensorflow会话
import tensorflow as tf
sess = tf.Session()
x = tf.Variable([1.0, 2.0])
b = tf.constant([3.0, 3.0])

tf.initialize_all_variables().run()
sud = tf.sub(x, b)
print sud.eval()
sess.close()
