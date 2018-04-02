import pandas as pd
import numpy as np
import tensorflow as tf

print("hello royston welcome to your first deep learning module")
x = tf.constant([5])
y = tf.constant([6])

result = tf.multiply(x,y)

print(result)
sess = tf.Session()
print("\n", sess.run(result))