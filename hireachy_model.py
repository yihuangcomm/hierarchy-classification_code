import numpy as np
import pandas
import os
import sys
import time
import shutil # we also need this
#it is very interesting
import tensorflow as tf

import logging

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import libmr
import math

#import read_data
def define_flags():
    flags = tf.app.flags
    flags.DEFINE_string("mode", "coarse_train", "coarse_train, coarse_test, fine1_train")  # MODE
    flags.DEFINE_string("model", "mlp", "Support mlp, cnn")    # MODEL

    flags.DEFINE_float("learning_rate", 0.0001, "Learning rate")   # LEARNING RATE
    flags.DEFINE_integer("epoch_number", 10000, "Number of epoches")
    flags.DEFINE_boolean("enable_bn", True, "Enable batch normalization")
    
    flags.DEFINE_string("train_file", "train_data/", "Train folder path")
    flags.DEFINE_string("dev_file", "dev_data/", "Validate folder path")
    flags.DEFINE_string("test_file", "test_data/",  "The test file contain other classes ")

    flags.DEFINE_string("output", "./io_hierarchy/io_coarse_mlp/0","Path for tensorboard")
    flags.DEFINE_string("checkpoint", "./checkpoint_hierarchy_mlp/0","Path for checkpoint")
    flags.DEFINE_integer("feature_index", "0","which feature we use")

    FLAGS = flags.FLAGS
    return FLAGS
# restore checkpoint function
def restore_from_checkpoint(sess, saver, checkpoint): 
    if checkpoint:        
        logging.info("Restore session from checkpoint: {}".format(checkpoint))        
        saver.restore(sess, checkpoint)       
        return True
    else:
        logging.warn("Checkpoint not found: {}".format(checkpoint))
        return False

def onehot(list):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(list)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded
def get_XY(Dir):
    X = []
    Y_coarse = []
    Y_fine = []
    for dirpath, _, filenames in os.walk(Dir):  
        for file in filenames :
            x = np.array(pandas.read_csv(os.path.join(dirpath, file)).loc[:])
            X.append(x)
            # depend on how to extract labels
            y_coarse = file[0:2]
            Y_coarse.append(y_coarse)
            y_fine = file[3:5]
            Y_fine.append(y_fine)
    Y_coarse = onehot(Y_coarse)
    Y_fine = onehot(Y_fine)
    X = np.array(X)
    return (X,Y_coarse,Y_fine)
# input the list of dir that should be processed
def get_XY_2(l_Dir):
    X = []
    Y = []
    for Dir in l_Dir:  
        for file in os.listdir(Dir) :
            x = np.array(pandas.read_csv(os.path.join(Dir, file)).loc[:])
            X.append(x)
            y = file[3:5]
            Y.append(y)
    Y = onehot(Y)
    X = np.array(X)
    return (X,Y)

def fc_layer(inputs, Weights, biases):
    Wx_plus_b = tf.matmul(inputs, Weights) + biases    
    return Wx_plus_b   

def batch_norm(inputs, is_train):
    scale = tf.Variable(tf.ones([inputs.shape[-1]]))
    shift = tf.Variable(tf.zeros([inputs.shape[-1]]))
    #axises = np.arange(len(inputs.shape)-1) # the dimension you wanna normalize, here [0] for batch
    axises = [0]
    batch_mean, batch_var = tf.nn.moments(inputs, axises)

    ema = tf.train.ExponentialMovingAverage(decay=0.5)
    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean,var = tf.cond(is_train, mean_var_with_update,lambda:(ema.average(batch_mean), ema.average(batch_var)))
    #mean, var = mean_var_with_update()
    epsilon = 0.001
    normed = tf.nn.batch_normalization(inputs, mean, var, shift, scale, epsilon)
    return normed 

#-----mlp----
def coarse_network(xs,is_train):  
    hidden_units = [300,100,50]
    weights = {
        'ful1': tf.Variable(tf.random_normal([input_units, hidden_units[0]]), name="fully_connect_weight1"),
        'ful2': tf.Variable(tf.random_normal([hidden_units[0], hidden_units[1]]), name="fully_connect_weight2"),
        'ful3': tf.Variable(tf.random_normal([hidden_units[1], output_units_coarse]), name="fully_connect_weight3")
      
    }
    biases = {
        'ful1': tf.Variable(tf.random_normal([hidden_units[0]]), name="fully_connect_bias1"),
        'ful2': tf.Variable(tf.random_normal([hidden_units[1]]), name="fully_connect_bias2"),
        'ful3': tf.Variable(tf.random_normal([output_units_coarse]), name="fully_connect_bias3")
        
    }

    #  LAYER STRUCTURE
    layer = fc_layer(xs, weights['ful1'],biases['ful1'])
    layer = batch_norm(layer,is_train)
    layer = tf.nn.relu(layer) 
     
    
    layer = fc_layer(layer, weights['ful2'],biases['ful2'])
    layer = batch_norm(layer,is_train)       
    coarse_layer = tf.nn.relu(layer)  

    layer = tf.cond(is_train, lambda:tf.nn.dropout(coarse_layer, 0.5),lambda:layer)
     
    layer = fc_layer(layer, weights['ful3'],biases['ful3'])
    layer = batch_norm(layer,is_train)
    
    return (layer,coarse_layer) 

def fine1_network(xs_fine1,is_train):  
    hidden_units = [50,30]

    weights = {
        'ful1': tf.Variable(tf.random_normal([input_units_fine, hidden_units[0]]), name="fine1_fully_connect_weight1"),
        'ful2': tf.Variable(tf.random_normal([hidden_units[0], hidden_units[1]]), name="fine1_fully_connect_weight2"),
        'ful3': tf.Variable(tf.random_normal([hidden_units[1], output_units_fine1]), name="fine1_fully_connect_weight3")
    }
    biases = {
        'ful1': tf.Variable(tf.random_normal([hidden_units[0]]), name="fine1_fully_connect_bias1"),
        'ful2': tf.Variable(tf.random_normal([hidden_units[1]]), name="fine1_fully_connect_bias2"),
        'ful3': tf.Variable(tf.random_normal([output_units_fine1]), name="fine1_fully_connect_bias3")
    }
    #  LAYER STRUCTURE
    layer = fc_layer(xs_fine1, weights['ful1'],biases['ful1'])
    layer = batch_norm(layer,is_train)  
    layer = tf.nn.relu(layer)
    layer = fc_layer(layer, weights['ful2'],biases['ful2']) 
    layer = batch_norm(layer,is_train)
    layer = tf.cond(is_train, lambda:tf.nn.dropout(layer, 0.5),lambda:layer)
    layer = fc_layer(layer, weights['ful3'],biases['ful3'])
    layer = batch_norm(layer,is_train) 
    return layer

def fine2_network(xs_fine2,is_train):  
    hidden_units = [50,30]

    weights = {
        'ful1': tf.Variable(tf.random_normal([input_units_fine, hidden_units[0]]), name="fine2_fully_connect_weight1"),
        'ful2': tf.Variable(tf.random_normal([hidden_units[0], hidden_units[1]]), name="fine2_fully_connect_weight2"),
        'ful3': tf.Variable(tf.random_normal([hidden_units[1], output_units_fine2]), name="fine2_fully_connect_weight3")
    }
    biases = {
        'ful1': tf.Variable(tf.random_normal([hidden_units[0]]), name="fine2_fully_connect_bias1"),
        'ful2': tf.Variable(tf.random_normal([hidden_units[1]]), name="fine2_fully_connect_bias2"),
        'ful3': tf.Variable(tf.random_normal([output_units_fine2]), name="fine2_fully_connect_bias3")
    }
    #  LAYER STRUCTURE
    layer = fc_layer(xs_fine2, weights['ful1'], biases['ful1'])
    layer = batch_norm(layer,is_train)  
    layer = tf.nn.relu(layer)
    layer = fc_layer(layer, weights['ful2'],biases['ful2']) 
    layer = batch_norm(layer,is_train)
    layer = tf.cond(is_train, lambda:tf.nn.dropout(layer, 0.5),lambda:layer)
    layer = fc_layer(layer, weights['ful3'],biases['ful3'])
    layer = batch_norm(layer,is_train) 
    return layer

def fine3_network(xs_fine3,is_train):  
    hidden_units = [50,30]

    weights = {
        'ful1': tf.Variable(tf.random_normal([input_units_fine, hidden_units[0]]), name="fine3_fully_connect_weight1"),
        'ful2': tf.Variable(tf.random_normal([hidden_units[0], hidden_units[1]]), name="fine3_fully_connect_weight2"),
        'ful3': tf.Variable(tf.random_normal([hidden_units[1], output_units_fine3]), name="fine3_fully_connect_weight3")
    }
    biases = {
        'ful1': tf.Variable(tf.random_normal([hidden_units[0]]), name="fine3_fully_connect_bias1"),
        'ful2': tf.Variable(tf.random_normal([hidden_units[1]]), name="fine3_fully_connect_bias2"),
        'ful3': tf.Variable(tf.random_normal([output_units_fine3]), name="fine3_fully_connect_bias3")
    }
    #  LAYER STRUCTURE
    layer = fc_layer(xs_fine3, weights['ful1'], biases['ful1'])
    layer = batch_norm(layer,is_train)  
    layer = tf.nn.relu(layer)
    layer = fc_layer(layer, weights['ful2'],biases['ful2']) 
    layer = batch_norm(layer,is_train)
    layer = tf.cond(is_train, lambda:tf.nn.dropout(layer, 0.5),lambda:layer)
    layer = fc_layer(layer, weights['ful3'],biases['ful3'])
    layer = batch_norm(layer,is_train) 
    return layer




logging.basicConfig(level=logging.INFO)
FLAGS = define_flags()
# HYPER PARAMETER
lr = FLAGS.learning_rate
training_iters = FLAGS.epoch_number
# dataset
train_dir = FLAGS.train_file
dev_dir = FLAGS.dev_file
test_dir = FLAGS.test_file
index = FLAGS.feature_index
mode = FLAGS.mode
model = FLAGS.model
checkpoint_path = FLAGS.checkpoint
output_path = FLAGS.output

ACTIVATION = tf.nn.relu
evaluate_step = 5

input_units = 500
input_units_fine = 100
output_units_coarse = 3
output_units_fine1 = 2
output_units_fine2 = 2
output_units_fine3 = 2
# for coarse classifier
x_train, y_train_coarse, y_train_fine = get_XY(train_dir)
x_dev, y_dev_coarse, y_dev_fine = get_XY(dev_dir)
x_test, y_test_coarse, y_test_fine = get_XY(test_dir)
x_train = x_train[:,:,index]
x_dev = x_dev[:,:,index]
x_test = x_test[:,:,index]
# Fine classifier 1
train_dir_fine1 = ['train_data/results_youtube','train_data/results_netflix']
dev_dir_fine1 = ['dev_data/results_youtube','dev_data/results_netflix']
test_dir_fine1 = ['test_data/results_youtube','test_data/results_netflix']
x_train_fine1, y_train_fine1 = get_XY_2(train_dir_fine1)
x_dev_fine1, y_dev_fine1 = get_XY_2(dev_dir_fine1)
x_test_fine1, y_test_fine1 = get_XY_2(test_dir_fine1)
x_train_fine1 = x_train_fine1[:,:,index]
x_dev_fine1 = x_dev_fine1[:,:,index]
x_test_fine1 = x_test_fine1[:,:,index]
# Fine classifier 2
train_dir_fine2 = ['train_data/results_spotify','train_data/results_xiami']
dev_dir_fine2 = ['dev_data/results_spotify','dev_data/results_xiami']
test_dir_fine2 = ['test_data/results_spotify','test_data/results_xiami']
x_train_fine2, y_train_fine2 = get_XY_2(train_dir_fine2)
x_dev_fine2, y_dev_fine2 = get_XY_2(dev_dir_fine2)
x_test_fine2, y_test_fine2 = get_XY_2(test_dir_fine2)
x_train_fine2 = x_train_fine2[:,:,index]
x_dev_fine2 = x_dev_fine2[:,:,index]
x_test_fine2 = x_test_fine2[:,:,index]
# Fine classifier 3
train_dir_fine3 = ['train_data/results_wiki','train_data/results_abc']
dev_dir_fine3 = ['dev_data/results_wiki','dev_data/results_abc']
test_dir_fine3 = ['test_data/results_wiki','test_data/results_abc']
x_train_fine3, y_train_fine3 = get_XY_2(train_dir_fine3)
x_dev_fine3, y_dev_fine3 = get_XY_2(dev_dir_fine3)
x_test_fine3, y_test_fine3 = get_XY_2(test_dir_fine3)
x_train_fine3 = x_train_fine3[:,:,index]
x_dev_fine3 = x_dev_fine3[:,:,index]
x_test_fine3 = x_test_fine3[:,:,index]

# X Y Placeholder
xs = tf.placeholder(tf.float32, [None, input_units], name = "xs_placeholder")
ys_coarse = tf.placeholder(tf.float32, [None, output_units_coarse], name = "ys_coarse_placeholder")
is_train = tf.placeholder(tf.bool, name = "is_train_placeholder")
# Fine classifier 1 
xs_fine1 = tf.placeholder(tf.float32, [None, input_units_fine], name = "xs_fine1_placeholder")
ys_fine1 = tf.placeholder(tf.float32, [None, output_units_fine1], name = "ys_fine1_placeholder")
# Fine classifier 2 
xs_fine2 = tf.placeholder(tf.float32, [None, input_units_fine], name = "xs_fine2_placeholder")
ys_fine2 = tf.placeholder(tf.float32, [None, output_units_fine2], name = "xs_fine2_placeholder")
# Fine classifier 3 
xs_fine3 = tf.placeholder(tf.float32, [None, input_units_fine], name = "xs_fine3_placeholder")
ys_fine3 = tf.placeholder(tf.float32, [None, output_units_fine3], name = "xs_fine3_placeholder")

# Define coarse classifer's loss, Acc
coarse_logits, coarse_layer = coarse_network(xs, is_train)
coarse_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = coarse_logits, labels = ys_coarse)
coarse_cost = tf.reduce_mean(coarse_cross_entropy)

coarse_softmax_pred = tf.nn.softmax(coarse_logits)
coarse_pred_index = tf.argmax(coarse_softmax_pred, 1)
coarse_y_index = tf.argmax(ys_coarse, 1)
coarse_correct_pred = tf.equal(coarse_pred_index, coarse_y_index)
coarse_accuracy = tf.reduce_mean(tf.cast(coarse_correct_pred, tf.float32))

coarse_optimizer = tf.train.AdamOptimizer(lr)
coarse_grads_and_vars = coarse_optimizer.compute_gradients(coarse_cost)
coarse_train_op = coarse_optimizer.apply_gradients(coarse_grads_and_vars)
# Define Fine1 classifer's loss, Acc
fine1_logits = fine1_network(xs_fine1, is_train)
fine1_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = fine1_logits, labels = ys_fine1)
fine1_cost = tf.reduce_mean(fine1_cross_entropy)
fine1_softmax_pred = tf.nn.softmax(fine1_logits)
fine1_pred_index = tf.argmax(fine1_softmax_pred, 1)
fine1_y_index = tf.argmax(ys_fine1, 1)
fine1_correct_pred = tf.equal(fine1_pred_index, fine1_y_index)
fine1_accuracy = tf.reduce_mean(tf.cast(fine1_correct_pred, tf.float32))
fine1_optimizer = tf.train.AdamOptimizer(lr)
fine1_grads_and_vars = fine1_optimizer.compute_gradients(fine1_cost)
fine1_train_op = fine1_optimizer.apply_gradients(fine1_grads_and_vars)
# Define Fine2 classifer's loss, Acc
fine2_logits = fine2_network(xs_fine2, is_train)
fine2_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = fine2_logits, labels = ys_fine2)
fine2_cost = tf.reduce_mean(fine2_cross_entropy)
fine2_softmax_pred = tf.nn.softmax(fine2_logits)
fine2_pred_index = tf.argmax(fine2_softmax_pred, 1)
fine2_y_index = tf.argmax(ys_fine2, 1)
fine2_correct_pred = tf.equal(fine2_pred_index, fine2_y_index)
fine2_accuracy = tf.reduce_mean(tf.cast(fine2_correct_pred, tf.float32))
fine2_optimizer = tf.train.AdamOptimizer(lr)
fine2_grads_and_vars = fine2_optimizer.compute_gradients(fine2_cost)
fine2_train_op = fine2_optimizer.apply_gradients(fine2_grads_and_vars)
# Define Fine3 classifer's loss, Acc
fine3_logits = fine3_network(xs_fine3, is_train)
fine3_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = fine3_logits, labels = ys_fine3)
fine3_cost = tf.reduce_mean(fine3_cross_entropy)
fine3_softmax_pred = tf.nn.softmax(fine3_logits)
fine3_pred_index = tf.argmax(fine3_softmax_pred, 1)
fine3_y_index = tf.argmax(ys_fine3, 1)
fine3_correct_pred = tf.equal(fine3_pred_index, fine3_y_index)
fine3_accuracy = tf.reduce_mean(tf.cast(fine3_correct_pred, tf.float32))
fine3_optimizer = tf.train.AdamOptimizer(lr)
fine3_grads_and_vars = fine3_optimizer.compute_gradients(fine3_cost)
fine3_train_op = fine3_optimizer.apply_gradients(fine3_grads_and_vars)

# coarse summary op
coarse_loss_summary = tf.summary.scalar("coarse_loss", coarse_cost)
coarse_acc_summary = tf.summary.scalar("coarse_accuracy", coarse_accuracy)
coarse_train_summary_op = tf.summary.merge([coarse_loss_summary, coarse_acc_summary])
coarse_dev_summary_op = tf.summary.merge([coarse_loss_summary, coarse_acc_summary])
# Fine1 summary op
fine1_loss_summary = tf.summary.scalar("fine1_loss", fine1_cost)
fine1_acc_summary = tf.summary.scalar("fine1_accuracy", fine1_accuracy)
fine1_train_summary_op = tf.summary.merge([fine1_loss_summary, fine1_acc_summary])
fine1_dev_summary_op = tf.summary.merge([fine1_loss_summary, fine1_acc_summary])
# Fine2 summary op
fine2_loss_summary = tf.summary.scalar("fine2_loss", fine2_cost)
fine2_acc_summary = tf.summary.scalar("fine2_accuracy", fine2_accuracy)
fine2_train_summary_op = tf.summary.merge([fine2_loss_summary, fine2_acc_summary])
fine2_dev_summary_op = tf.summary.merge([fine2_loss_summary, fine2_acc_summary])
# Fine3 summary op
fine3_loss_summary = tf.summary.scalar("fine3_loss", fine3_cost)
fine3_acc_summary = tf.summary.scalar("fine3_accuracy", fine3_accuracy)
fine3_train_summary_op = tf.summary.merge([fine3_loss_summary, fine3_acc_summary])
fine3_dev_summary_op = tf.summary.merge([fine3_loss_summary, fine3_acc_summary])

timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, output_path, timestamp))
coarse_train_summary_dir = os.path.join(out_dir, "summaries", "coarse_train")
coarse_dev_summary_dir = os.path.join(out_dir, "summaries", "coarse_dev")
fine1_train_summary_dir = os.path.join(out_dir, "summaries", "fine1_train")
fine1_dev_summary_dir = os.path.join(out_dir, "summaries", "fine1_dev")
fine2_train_summary_dir = os.path.join(out_dir, "summaries", "fine2_train")
fine2_dev_summary_dir = os.path.join(out_dir, "summaries", "fine2_dev")
fine3_train_summary_dir = os.path.join(out_dir, "summaries", "fine3_train")
fine3_dev_summary_dir = os.path.join(out_dir, "summaries", "fine3_dev")

# checkpoint path
if os.path.exists(checkpoint_path) == False:
        os.makedirs(checkpoint_path)
CHECKPOINT_FILE = checkpoint_path + "/checkpoint.ckpt"
LATEST_CHECKPOINT = tf.train.latest_checkpoint(checkpoint_path)
 
# Define global step 
coarse_step_value = tf.Variable(0, trainable=False, name='coarse_global_step')
coarse_global_step = tf.assign_add(coarse_step_value, 1,  name='coarse_increment_global_step')
fine1_step_value = tf.Variable(0, trainable=False, name='fine1_global_step')
fine1_global_step = tf.assign_add(fine1_step_value, 1,  name='fine1_increment_global_step')
fine2_step_value = tf.Variable(0, trainable=False, name='fine2_global_step')
fine2_global_step = tf.assign_add(fine2_step_value, 1,  name='fine2_increment_global_step')
fine3_step_value = tf.Variable(0, trainable=False, name='fine3_global_step')
fine3_global_step = tf.assign_add(fine3_step_value, 1,  name='fine3_increment_global_step')
# Define best validation  
coarse_best_validation = tf.Variable(0, trainable=False, name='coarse_best_validation_accuracy')
fine1_best_validation = tf.Variable(0, trainable=False, name='fine1_best_validation_accuracy')
fine2_best_validation = tf.Variable(0, trainable=False, name='fine2_best_validation_accuracy')
fine3_best_validation = tf.Variable(0, trainable=False, name='fine3_best_validation_accuracy')
# Initialize saver
saver = tf.train.Saver()
# Define init op
init_op = tf.initialize_all_variables()

#------CREATE SESSION TO RUN--------#
with tf.Session() as sess:  
   
    sess.run(init_op) 
    coarse_train_summary_writer = tf.summary.FileWriter(coarse_train_summary_dir, sess.graph)
    coarse_dev_summary_writer = tf.summary.FileWriter(coarse_dev_summary_dir, sess.graph)
    fine1_train_summary_writer = tf.summary.FileWriter(fine1_train_summary_dir, sess.graph)
    fine1_dev_summary_writer = tf.summary.FileWriter(fine1_dev_summary_dir, sess.graph)
    fine2_train_summary_writer = tf.summary.FileWriter(fine2_train_summary_dir, sess.graph)
    fine2_dev_summary_writer = tf.summary.FileWriter(fine2_dev_summary_dir, sess.graph)
    fine3_train_summary_writer = tf.summary.FileWriter(fine3_train_summary_dir, sess.graph)
    fine3_dev_summary_writer = tf.summary.FileWriter(fine3_dev_summary_dir, sess.graph)

    if mode == 'coarse_train':
        restore_from_checkpoint(sess, saver, LATEST_CHECKPOINT)
        coarse_best_validation_accuracy = sess.run(coarse_best_validation)
       
        for i in range(training_iters):          
            _,summaries, loss, acc, step= sess.run(
                [coarse_train_op, coarse_train_summary_op, coarse_cost, coarse_accuracy, coarse_global_step],
                feed_dict = {
                    xs: x_train,
                    ys_coarse: y_train_coarse,
                    is_train: True    
                }
            )
            print("step=" +"{}".format(step)+",loss=" +"{:.4f}".format(loss) + 
                ", Training Accuracy= "+"{:.5f}".format(acc))
            coarse_train_summary_writer.add_summary(summaries,step)
            
            summaries, loss, acc = sess.run(
                [coarse_dev_summary_op, coarse_cost, coarse_accuracy],
                feed_dict = {
                    xs: x_dev,
                    ys_coarse: y_dev_coarse,
                    is_train: False                
                }
            )
            print("step=" +"{}".format(step)+", loss=" +
                    "{:.4f}".format(loss) + ", Dev Accuracy= " +"{:.5f}".format(acc))
            coarse_dev_summary_writer.add_summary(summaries,step)
            # save the best checkpoint
            if (acc > coarse_best_validation_accuracy):
                coarse_best_validation_accuracy = acc
                sess.run(tf.assign(coarse_best_validation,coarse_best_validation_accuracy))
                saver.save(sess, CHECKPOINT_FILE, global_step=step)

    if mode == 'coarse_test':
        if restore_from_checkpoint(sess, saver, LATEST_CHECKPOINT) == False:  
            sys.exit()      
            logging.error("No checkpoint for inferencing, exit now")
            exit(1)
        acc, loss = sess.run(
            [coarse_accuracy, coarse_cost],
            feed_dict={                
                xs: x_test,
                ys_coarse: y_test_coarse,
                is_train: False            
            })    
        print("loss=" +"{:.4f}".format(loss) + ", Testing Accuracy= " +"{:.5f}".format(acc))
    
    if mode == 'fine1_train':
        restore_from_checkpoint(sess, saver, LATEST_CHECKPOINT)
        fine1_best_validation_accuracy = sess.run(fine1_best_validation)

        x_coarse = sess.run(
                coarse_layer,
                feed_dict = {
                    xs: x_train_fine1,
                    is_train: True    
                }
            )
        x_coarse_dev = sess.run(
                coarse_layer,
                feed_dict = {
                    xs: x_dev_fine1,
                    is_train: False    
                }
            )
        for i in range(training_iters):          
            _, summaries, loss, acc, step = sess.run(
                [fine1_train_op,fine1_train_summary_op,fine1_cost,fine1_accuracy,fine1_global_step],
                feed_dict = {
                    xs_fine1: x_coarse,
                    ys_fine1: y_train_fine1,
                    is_train: True    
                }
            )
            print("step=" +"{}".format(step)+", loss=" +"{:.4f}".format(loss) + ", Training Accuracy= "+"{:.5f}".format(acc))            
            fine1_train_summary_writer.add_summary(summaries,step)

            loss, acc = sess.run(
                [fine1_cost, fine1_accuracy],
                feed_dict = {
                    xs_fine1: x_coarse_dev,
                    ys_fine1: y_dev_fine1,
                    is_train: False                
                }
            )
            print("loss=" + "{:.4f}".format(loss) + ", Dev Accuracy= " +"{:.5f}".format(acc))
            fine1_dev_summary_writer.add_summary(summaries,step)
            # save the best checkpoint
            if (acc > fine1_best_validation_accuracy):
                fine1_best_validation_accuracy = acc
                sess.run(tf.assign(fine1_best_validation,fine1_best_validation_accuracy))
                saver.save(sess, CHECKPOINT_FILE, global_step=step)
    
    if mode == 'fine2_train':
        restore_from_checkpoint(sess, saver, LATEST_CHECKPOINT)
        fine2_best_validation_accuracy = sess.run(fine2_best_validation)

        x_coarse = sess.run(
                coarse_layer,
                feed_dict = {
                    xs: x_train_fine2,
                    is_train: True    
                }
            )
        x_coarse_dev = sess.run(
                coarse_layer,
                feed_dict = {
                    xs: x_dev_fine2,
                    is_train: False    
                }
            )

        for i in range(training_iters): 
            _, summaries, loss, acc, step = sess.run(
                [fine2_train_op,fine2_train_summary_op,fine2_cost,fine2_accuracy,fine2_global_step],
                feed_dict = {
                    xs_fine2: x_coarse,
                    ys_fine2: y_train_fine2,
                    is_train: True    
                }
            )
            print("step=" +"{}".format(step)+", loss=" +"{:.4f}".format(loss) + ", Training Accuracy= "+"{:.5f}".format(acc))            
            fine2_train_summary_writer.add_summary(summaries,step)

            loss, acc = sess.run(
                [fine2_cost, fine2_accuracy],
                feed_dict = {
                    xs_fine2: x_coarse_dev,
                    ys_fine2: y_dev_fine2,
                    is_train: False                
                }
            )
            print("loss=" + "{:.4f}".format(loss) + ", Dev Accuracy= " +"{:.5f}".format(acc))
            fine2_dev_summary_writer.add_summary(summaries,step)
            # save the best checkpoint
            if (acc > fine2_best_validation_accuracy):
                fine2_best_validation_accuracy = acc
                sess.run(tf.assign(fine2_best_validation,fine2_best_validation_accuracy))
                saver.save(sess, CHECKPOINT_FILE, global_step=step)
    
    if mode == 'fine3_train':
        restore_from_checkpoint(sess, saver, LATEST_CHECKPOINT)
        fine3_best_validation_accuracy = sess.run(fine3_best_validation)

        x_coarse = sess.run(
                coarse_layer,
                feed_dict = {
                    xs: x_train_fine3,
                    is_train: True    
                }
            )
        x_coarse_dev = sess.run(
                coarse_layer,
                feed_dict = {
                    xs: x_dev_fine3,
                    is_train: False    
                }
            )

        for i in range(training_iters): 
            _, summaries, loss, acc, step = sess.run(
                [fine3_train_op,fine3_train_summary_op,fine3_cost,fine3_accuracy,fine3_global_step],
                feed_dict = {
                    xs_fine3: x_coarse,
                    ys_fine3: y_train_fine3,
                    is_train: True    
                }
            )
            print("step=" +"{}".format(step)+", loss=" +"{:.4f}".format(loss) + ", Training Accuracy= "+"{:.5f}".format(acc))            
            fine3_train_summary_writer.add_summary(summaries,step)

            loss, acc = sess.run(
                [fine3_cost, fine3_accuracy],
                feed_dict = {
                    xs_fine3: x_coarse_dev,
                    ys_fine3: y_dev_fine3,
                    is_train: False                
                }
            )
            print("loss=" + "{:.4f}".format(loss) + ", Dev Accuracy= " +"{:.5f}".format(acc))
            fine3_dev_summary_writer.add_summary(summaries,step)
            # save the best checkpoint
            if (acc > fine3_best_validation_accuracy):
                fine3_best_validation_accuracy = acc
                sess.run(tf.assign(fine3_best_validation,fine3_best_validation_accuracy))
                saver.save(sess, CHECKPOINT_FILE, global_step=step)
    
    if mode == 'test':
        restore_from_checkpoint(sess, saver, LATEST_CHECKPOINT)
        x_coarse, softmax, index = sess.run(
                [coarse_layer,coarse_softmax_pred,coarse_pred_index],
                feed_dict = {
                    xs: x_train,
                    is_train: False    
                }
            )
        index_0 = []
        index_1 = []
        index_2 = []
        for i in range(len(index)):
            if index[i] == 0:
                index_0.append(i)
            elif index[i] == 1:
                index_1.append(i)
            elif index[i] == 2:
                index_2.append(i)
            else:
                pass
        x_fine1 = x_coarse[index_0]
        x_fine2 = x_coarse[index_1]
        x_fine3 = x_coarse[index_2]
        
        fine1_index = sess.run(
                [fine1_pred_index],
                feed_dict = {
                    xs_fine1: x_fine1,
                    is_train: False    
                }
            )
        fine2_index = sess.run(
                [fine2_pred_index],
                feed_dict = {
                    xs_fine2: x_fine2,
                    is_train: False    
                }
            )
        fine3_index = sess.run(
                [fine3_pred_index],
                feed_dict = {
                    xs_fine3: x_fine3,
                    is_train: False    
                }
            )

    
    
    
    
    # given the final class 
    # if mode == 'test':
    #     restore_from_checkpoint(sess, saver, LATEST_CHECKPOINT)
    #     x_coarse, softmax, index = sess.run(
    #             [coarse_layer,coarse_softmax_pred,coarse_pred_index],
    #             feed_dict = {
    #                 xs: x_train,
    #                 is_train: False    
    #             }
    #         )
    #     index_0 = []
    #     index_1 = []
    #     index_2 = []
    #     for i in range(len(index)):
    #         if index[i] == 0:
    #             index_0.append(i)
    #         elif index[i] == 1:
    #             index_1.append(i)
    #         elif index[i] == 2:
    #             index_2.append(i)
    #         else:
    #             pass
    #     x_fine1 = x_coarse[index_0]
    #     x_fine2 = x_coarse[index_1]
    #     x_fine3 = x_coarse[index_2]
        
    #     fine1_index = sess.run(
    #             [fine1_pred_index],
    #             feed_dict = {
    #                 xs_fine1: x_fine1,
    #                 is_train: False    
    #             }
    #         )
    #     fine2_index = sess.run(
    #             [fine2_pred_index],
    #             feed_dict = {
    #                 xs_fine2: x_fine2,
    #                 is_train: False    
    #             }
    #         )
    #     fine3_index = sess.run(
    #             [fine3_pred_index],
    #             feed_dict = {
    #                 xs_fine3: x_fine3,
    #                 is_train: False    
    #             }
    #         )


        
