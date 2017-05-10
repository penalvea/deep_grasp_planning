import tensorflow as tf
import numpy as np
import time
import datetime
import read_dataset_2d_object
import matplotlib.pyplot as plt

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

height, width, depth= 60, 60, 60

dataset_path="/home/penalvea/dataset2/geometrics"

folder="/first"

iterations_next_folder=500

iterations_complete=99999
run_until=datetime.datetime(2018, 10, 29, 13, 30)

output_path="/home/penalvea/NetResults/"

batch_size=100.0



def weight_variable(shape):

  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)



def convPoolLayer_Red(inp, size, layersIn, layersOut,sizeRed):
    w_conv=weight_variable([size, size, layersIn, layersOut])
    b_conv=bias_variable([layersOut])

    conv=tf.nn.relu(tf.nn.conv2d(inp, w_conv, strides=[1,1,1,1], padding="SAME")+b_conv)

    pool=tf.nn.max_pool(conv, ksize=[1, sizeRed, sizeRed, 1], strides=[1, sizeRed, sizeRed, 1], padding="SAME")

    return pool


def convPoolLayer(inp, size, layersIn, layersOut):
    w_conv = weight_variable([size, size, layersIn, layersOut])
    b_conv = bias_variable([layersOut])

    conv = tf.nn.relu(tf.nn.conv2d(inp, w_conv, strides=[1, 1, 1, 1], padding="SAME") + b_conv)
    return conv


def denselyConnLayer(inp, layersIn, layersOut):
    w_conv = weight_variable([layersIn, layersOut])
    b_conv = bias_variable([layersOut])

    dense=tf.nn.relu(tf.matmul(inp, w_conv)+b_conv)

    return dense


def denselyConnLayerLineal(inp, layersIn, layersOut):
    w_conv = weight_variable([layersIn, layersOut])
    b_conv = bias_variable([layersOut])

    dense = tf.matmul(inp, w_conv) + b_conv
    return dense


def inference_4layers(inp):
    conv1=convPoolLayer_Red(inp, 5, 1, 64, 5)
    print conv1
    conv2=convPoolLayer_Red(conv1, 3, 64, 128, 3)
    print conv2


    flat=tf.reshape(conv2, [-1,4*4*128])
    dens1=denselyConnLayer(flat, 4*4*128, 2048)
    dens2=denselyConnLayerLineal(dens1, 2048, 4)

    return dens2





[training, validation]=read_dataset_2d_object.readNextDataSet(dataset_path, folder, height, width, depth)

print(validation)


train_objects=training.num_examples
val_objects=validation.num_examples


sess=tf.InteractiveSession()
x=tf.placeholder("float", [None, height, width, 1])

x_input=tf.reshape(x, [-1, height, width, 1])
y_=tf.placeholder("float", shape=[None, 4])


res=inference_4layers(x_input)



loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=res))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss_function)
correct_prediction = tf.equal(tf.argmax(res, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))





saver = tf.train.Saver()

tf.global_variables_initializer().run()




### Train party hard

acum=0
count=0
i=0
epochs=0
start_time=init_time=time.time()
now=datetime.datetime.now()
bad=0
unos = 0
while(now<run_until):
    batch=training.next_batch(batch_size)





    _ , output, label, result, correct_pred, accu=sess.run([train_step, loss_function, y_, res, correct_prediction, accuracy], feed_dict={x: batch[0], y_: batch[1]})
    print (output)
    print (correct_pred)
    print (accu)

