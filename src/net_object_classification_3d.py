import tensorflow as tf
import numpy as np
import time
import datetime
import read_dataset_3d_object
import matplotlib.pyplot as plt
import os.path

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

height, width, depth= 30, 30, 30

dataset_path="/home/penalvea/hollows/geometrics"

write_objects1="/home/penalvea/hollows/first"
ready_objects1="/home/penalvea/hollows/first_ready"
write_objects2="/home/penalvea/dataset2/second"
ready_objects2="/home/penalvea/dataset2/second_ready"
write_objects3="/home/penalvea/dataset2/third"
ready_objects3="/home/penalvea/dataset2/third_ready"
write_objects4="/home/penalvea/dataset2/fourth"
ready_objects4="/home/penalvea/dataset2/fourth_ready"
write_objects5="/home/penalvea/dataset2/fifth"
ready_objects5="/home/penalvea/dataset2/fifth_ready"
write_objects6="/home/penalvea/dataset2/sixth"
ready_objects6="/home/penalvea/dataset2/sixth_ready"
write_objects7="/home/penalvea/dataset2/seventh"
ready_objects7="/home/penalvea/dataset2/seventh_ready"
write_objects8="/home/penalvea/dataset2/eighth"
ready_objects8="/home/penalvea/dataset2/eighth_ready"
write_objects9="/home/penalvea/dataset2/nineth"
ready_objects9="/home/penalvea/dataset2/nineth_ready"
write_objects10="/home/penalvea/dataset2/tenth"
ready_objects10="/home/penalvea/dataset2/tenth_ready"


folder1="/first"
folder2="/second"
folder3="/third"
folder4="/fourth"
folder5="/fifth"
folder6="/sixth"
folder7="/seventh"
folder8="/eighth"
folder9="/nineth"
folder10="/tenth"



iterations_next_folder=50

iterations_complete=99999
run_until=datetime.datetime(2018, 10, 29, 13, 30)

output_path="/home/penalvea/NetResults/hollows/"

batch_size=4.0



def weight_variable(shape):

  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)



def convPoolLayer_Red(inp, size, layersIn, layersOut,sizeRed, name):
    with tf.name_scope(name) as scope:
        w_conv=weight_variable([size, size, size, layersIn, layersOut])
        b_conv=bias_variable([layersOut])

        conv=tf.nn.relu(tf.nn.conv3d(inp, w_conv, strides=[1,1,1,1,1], padding="SAME")+b_conv)
        tf.summary.histogram(name + "/Filters", w_conv)

        pool=tf.nn.max_pool3d(conv, ksize=[1, sizeRed, sizeRed, sizeRed, 1], strides=[1, sizeRed, sizeRed, sizeRed, 1], padding="SAME")

    return pool


def convPoolLayer(inp, size, layersIn, layersOut, name):
    with tf.name_scope(name) as scope:
        w_conv = weight_variable([size, size, size, layersIn, layersOut])
        b_conv = bias_variable([layersOut])
        tf.summary.histogram(name + "/Filters", w_conv)
        conv = tf.nn.relu(tf.nn.conv3d(inp, w_conv, strides=[1, 1, 1, 1, 1], padding="SAME") + b_conv)
    return conv


def denselyConnLayer(inp, layersIn, layersOut, name):
    with tf.name_scope(name) as scope:
        w_conv = weight_variable([layersIn, layersOut])
        b_conv = bias_variable([layersOut])
        tf.summary.histogram(name + "/Filters", w_conv)
        dense=tf.nn.relu(tf.matmul(inp, w_conv)+b_conv)
        dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=True)

    return dropout


def denselyConnLayerLineal(inp, layersIn, layersOut, name):
    with tf.name_scope(name) as scope:
        w_conv = weight_variable([layersIn, layersOut])
        b_conv = bias_variable([layersOut])
        tf.summary.histogram(name + "/Filters", w_conv)

        dense = tf.matmul(inp, w_conv) + b_conv
    return dense


def inference_4layers(inp):
    conv1=convPoolLayer_Red(inp, 5, 1, 4, 5, 'conv1_class')
    print conv1
    conv2=convPoolLayer_Red(conv1, 3, 4, 8, 3, 'conv2_class')
    print conv2
    conv3 = convPoolLayer(conv2, 3, 8, 12, 'conv3_class')
    print conv3
    #conv4 = convPoolLayer(conv3, 3, 80, 160)
    #print conv4


    flat=tf.reshape(conv3, [-1,2*2*2*12])
    dens1=denselyConnLayer(flat, 2*2*2*12, 20, 'dense1_class')
    lineal1=denselyConnLayerLineal(dens1, 20, 4, 'lineal1_class')

    return lineal1








sess=tf.InteractiveSession()
x=tf.placeholder("float", [None, height, width, depth, 1])

x_input=tf.reshape(x, [-1, height, width, depth, 1])
y_=tf.placeholder("float", shape=[None, 4])


res=inference_4layers(x_input)


with tf.name_scope('cross_entropy'):

    #loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=res))
    loss_function = tf.losses.softmax_cross_entropy(onehot_labels=y_, logits=res)

#loss_function = -tf.reduce_sum(y_*tf.log(tf.nn.softmax(res) + 1e-10))

with tf.name_scope('train'):
    train_step = tf.contrib.layers.optimize_loss(loss_function, tf.contrib.framework.get_global_step(), optimizer='Adam', learning_rate=0.001)

epoch_loss=tf.Variable(1.0, name="epochLoss")
tf.summary.scalar("epochLoss", epoch_loss)

epoch_accuracy=tf.Variable(0.0, name="epochAccuracy")
tf.summary.scalar("epochAccuracy", epoch_accuracy)

# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss_function)

#train_step = tf.train.AdamOptimizer(1e-4).minimize(loss_function)
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(res, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))





merged=tf.summary.merge_all()
train_writer=tf.summary.FileWriter("/home/penalvea/tensorboard/hollow/train", sess.graph)
test_writer=tf.summary.FileWriter("/home/penalvea/tensorboard/hollow/test")

saver = tf.train.Saver()

tf.global_variables_initializer().run()




### Train party hard

acum=0
count=0
i=0
epochs=0
start_time=init_time=time.time()
now=datetime.datetime.now()


predictions=[]


start_time = init_time = time.time()
now=datetime.datetime.now()
change=1
folder=1
objects=0
best = 100.0
while(now<run_until):
    if change==1:
        if folder==1:
	    print (ready_objects1)
            while not os.path.isfile(ready_objects1):
              time.sleep(1)
            #os.remove(ready_objects1)

            [training, validation] = read_dataset_3d_object.readNextDataSet(dataset_path, folder1, height, width, depth)
            open(write_objects1, 'a').close()
            folder=2



        train_objects = training.num_examples
        val_objects = validation.num_examples
        change=0


    batch=training.next_batch(batch_size)

    #print (batch[1])
    objects+=len(batch[0])
    _ , output, label, result, correct_pred, summary=sess.run([train_step, loss_function, y_, res, correct_prediction, merged], feed_dict={x: batch[0], y_: batch[1]})
    acum+=output
    predictions.extend(correct_pred)

    #print (label)

    if i>0 and i%int(train_objects/batch_size)==0:


        new_time = time.time()

        accuracy=np.mean(correct_pred,  dtype=np.float64)
	
        sess.run(epoch_loss.assign(acum/objects))
        sess.run(epoch_accuracy.assign(accuracy))
	
        epochs+=1
        train_writer.add_summary(summary, epochs)
	

        print ("iteration %d, loss_function: %.6f, correct_predictions: %f, elapsed time: %.2f, iteration time: %.2f" % (i/int(objects/batch_size) , acum/objects,  accuracy, (new_time-init_time)/60,  (new_time-start_time)/60))
        start_time = time.time()
        acum=0
        objects=0
       
	
        if epochs%10==0:
            acum_test=0
            objects_test=0
            predictions_test=[]
            for j in range(int(val_objects/batch_size)):
                batch_test=validation.next_batch(batch_size)
                loss_test, acc_test, summary=sess.run([loss_function, correct_prediction, merged], feed_dict={x: batch_test[0],  y_:batch_test[1]})
                objects_test+=len(batch_test[0])
                acum_test+=loss_test
                predictions_test.extend(acc_test)
            accuracy_test=np.mean(predictions_test, dtype=np.float64)
            sess.run(epoch_loss.assign(acum_test/ objects_test))
            sess.run(epoch_accuracy.assign(accuracy_test))
            test_writer.add_summary(summary, epochs)
            print ("Test:    iteration %d, loss_function: %.6f, correct_predictions: %f" % (epochs, acum_test/objects_test, accuracy_test))
	    if best>(acum_test/objects_test):
                best=(acum_test/objects_test)
                saver.save(sess, output_path + "modelckp" + str(epochs) + ".ckpt")
    i = i + 1
    now = datetime.datetime.now()
