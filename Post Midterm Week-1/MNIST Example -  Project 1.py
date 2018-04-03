import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("tmp/data/", one_hot=True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

# The picture taken in is 28x28 pixels. Matrix dimensions are height x width
x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float")

def neural_network_model(data):
    hidden_1_layer  = {"weights": tf.Variable(tf.random_normal([784,n_nodes_hl1]))}