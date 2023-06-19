import tensorflow as tf

print(tf.__version__)

rand = tf.random.uniform([2], 1, 2)
print(rand)