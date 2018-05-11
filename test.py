'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


def add_layer(inputs, in_size, out_size, activation_function = None):
    with tf.name_scope("layers"):
        with tf.name_scope("weights"):
            w = tf.Variable(tf.random_normal([in_size,out_size]),name="W")
        with tf.name_scope("bias"):
            b = tf.Variable(tf.zeros([1,out_size])+0.01,name="B")
        with tf.name_scope("input"):
            wx_plus_b = tf.matmul(inputs,w) + b
        if activation_function is None:
            output = wx_plus_b
        else:
            output = activation_function(wx_plus_b)
        return output

#x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
#nois_data = np.random.normal(0,0.05,x_data.shape).astype(np.float32)
#y_data = np.square(x_data) - 0.5 + nois_data
#plt.scatter(x_data,y_data)
#plt.show()

#xs = tf.placeholder(tf.float32,[None,1])
#ys = tf.placeholder(tf.float32,[None,1])

xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,10])

prediction = add_layer(xs,784,10,activation_function=tf.nn.softmax)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)





#layer1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
#prediction = add_layer(layer1,10,1,activation_function=None)
#with tf.name_scope("loss"):
#    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]),name="RRR")

#with tf.name_scope("train"):
#    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()



sess = tf.Session()
#writer = tf.summary.FileWriter("logs/",sess.graph)
sess.run(init)
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})

    if i%50 ==0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels))


sess.close()





input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1,input2)
with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[7.],input2:[5.]}))






state = tf.Variable(0,name='counter')
print(state)
one = tf.constant(1,name='constant1')
new_vaule = tf.add(state,one)
print(new_vaule)
update = tf.assign(state,new_vaule)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(2):
        sess.run(update)
        print(sess.run(state))





matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])
product = tf.matmul(matrix1,matrix2)
sess1 = tf.Session()
result = sess1.run(product)
print(result)
sess1.close()

with tf.Session() as ssee2:
    result2 = ssee2.run(product)
    print(result2)





x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 +0.3

weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))


y = weights*x_data +biases

loss = tf.reduce_mean(tf.square(y-y_data))
optimier = tf.train.GradientDescentOptimizer(0.3)
train = optimier.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(500):
    sess.run(train)
    if step % 5 == 0:
        print(step, sess.run(weights), sess.run(biases))





'''
import sys
import numpy as np
import os
from xml.dom.minidom import parse
import xml.dom.minidom
import networkx as nx

#read infor form xml files and output graphs
#each mesh is a graph
def load_data(model_xml_str):

    DOMTree = xml.dom.minidom.parse(model_xml_str)
    collection = DOMTree.documentElement
    nodes = collection.getElementsByTagName("node")
    edges = collection.getElementsByTagName("edge")
    groups = collection.getElementsByTagName("group")
    for node in nodes:
        print(node.getElementsByTagName('mesh')[0].childNodes[0].data)


# 读取model名称
modelName = np.genfromtxt('D:\\langxf\\0416\\_chair_shapes.txt',dtype=np.dtype(str))
models = modelName[:, 1]



for i in range(len(models)):
    f = "D:\\langxf\\0416\\chair\\{}\\{}.xml".format(models[i],models[i])
    load_data(f)

    # read feature
    f = "D:\\langxf\\0416\\chair\\{}\\".format(models[i],models[i])

