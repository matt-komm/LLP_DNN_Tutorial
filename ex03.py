import tensorflow as tf
import numpy as np
import keras


#initialize input as a placeholder
x = tf.placeholder("float", [None, 10])

keras_input = keras.layers.Input(tensor=x)
out = keras.layers.Dense(
    20, 
    activation="tanh", 
    use_bias=True, 
    kernel_initializer='random_normal', 
    bias_initializer='random_normal'
)(keras_input)
out = keras.layers.Dense(
    20, 
    activation="tanh", 
    use_bias=True, 
    kernel_initializer='random_normal', 
    bias_initializer='random_normal'
)(out)
out = keras.layers.Dense(
    1, 
    activation="tanh", 
    use_bias=True, 
    kernel_initializer='random_normal', 
    bias_initializer='random_normal'
)(out)


init_op = tf.global_variables_initializer() #operation to call random initializers

sess = tf.Session()
sess.run(init_op) #execute initialization

#use keras model class to store the weights
model = keras.Model(inputs=[keras_input], outputs=[out])
model.save_weights("model_ex03.hdf5")

#calculate the output
print sess.run(
    out,
    feed_dict={x:[np.linspace(1,10,num=10)]} #need to feed input to x since placeholder
)

