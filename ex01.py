import tensorflow as tf
import numpy as np

#initialize constant tensors
x = tf.convert_to_tensor(
    np.linspace(1,10,num=10),
    dtype=tf.float32
)
y = tf.convert_to_tensor(
    np.linspace(11,20,num=10),
    dtype=tf.float32
)
print "tensor x:",x
print "tensor y:",y

#make a calculation
z = tf.add(x,y)
print "tensor z:",z

sess = tf.Session()
print "result z:",sess.run(z)



