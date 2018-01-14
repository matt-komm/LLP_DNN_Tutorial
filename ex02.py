import tensorflow as tf
import numpy as np

weights = []
biases = []
def make_layer(input_tensor,length):
    #declare weights/biases as variable -> to be learned during training
    weight = tf.Variable(
        tf.random_normal([input_tensor.shape[1].value,length])
    )
    weights.append(weight)
    bias = tf.Variable(
        tf.random_normal([length])
    )
    biases.append(bias)
    result = tf.matmul(input_tensor,weight) #matrix multiplication
    result = tf.add(result,bias) #add bias vector
    result = tf.tanh(result) #apply activation function element-wise
    return result

#initialize input as a placeholder
x = tf.placeholder("float", [None, 10])

out = make_layer(x,20) #20 hidden layers
out = make_layer(out,20) #20 hidden layers
out = make_layer(out,1) #1 output layer


init_op = tf.global_variables_initializer() #operation to call random initializers

#Advanced: for calculating the gradients
#grad = tf.gradients(out,weights+biases)

sess = tf.Session()
sess.run(init_op) #execute initialization
#calculate the output
print sess.run(
    out,
    feed_dict={x:[np.linspace(1,10,num=10)]} #need to feed input to x since placeholder
)

#Advanced: calculates the gradients wrt the weight & bias variables
#print sess.run(
#    grad,
#    feed_dict={x:[np.linspace(1,10,num=10)]} #need to feed input to x since placeholder
#)


