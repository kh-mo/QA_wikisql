import os
import sys
sys.path.append(os.getcwd() + "/image classification/")
import Block as B
import numpy as np
import tensorflow as tf

class Basic():
    ## 네트워크 구조 지정
    x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
    y = tf.placeholder(tf.float32, shape=[None, 2])
    bol = tf.placeholder(tf.bool)

    conv1 = B.Conv(input=x, filter=[11,11,3], output_channel=96, stride=4, init="xavier", trainable=True, name="conv1")
    batch1 = B.BN(input=conv1, trainable=bol, name="batch1")
    act1 = B.Act(input=batch1, function="relu", name="relu1")
    pool1 = B.Pool(input=act1, filter=2, stride=2, function="max", name="pool1")

    conv2 = B.Conv(input=pool1, filter=[5,5,96], output_channel=256, stride=1, init="xavier", trainable=True, name="conv2")
    batch2 = B.BN(input=conv2, trainable=bol, name="batch2")
    act2 = B.Act(input=batch2, function="relu", name="relu2")
    pool2 = B.Pool(input=act2, filter=2, stride=2, function="max", name="pool2")

    conv3 = B.Conv(input=pool2, filter=[3,3,256], output_channel=384, stride=1, init="xavier", trainable=True, name="conv3")
    batch3 = B.BN(input=conv3, trainable=bol, name="batch3")
    act3 = B.Act(input=batch3, function="relu", name="relu3")

    conv4 = B.Conv(input=act3, filter=[3,3,384], output_channel=384, stride=1, init="xavier", trainable=True, name="conv4")
    batch4 = B.BN(input=conv4, trainable=bol, name="batch4")
    act4 = B.Act(input=batch4, function="relu", name="relu4")

    conv5 = B.Conv(input=act4, filter=[3,3,384], output_channel=384, stride=1, init="xavier", trainable=True, name="conv5")
    batch5 = B.BN(input=conv5, trainable=bol, name="batch5")
    act5 = B.Act(input=batch5, function="relu", name="relu5")
    pool3 = B.Pool(input=act5, filter=2, stride=2, function="max", name="pool5")

    fc1 = B.FC(input=pool3, output_channel=2048, init="xavier", trainable=True, name="fc1")
    batch6 = B.BN(input=fc1, trainable=bol, name="batch6")
    act6 = B.Act(input=batch6, function="relu", name="relu6")

    fc2 = B.FC(input=act6, output_channel=2048, init="xavier", trainable=True, name="fc2")
    batch7 = B.BN(input=fc2, trainable=bol, name="batch7")
    act7 = B.Act(input=batch7, function="relu", name="relu7")

    fc3 = B.FC(input=act7, output_channel=2, init="xavier", trainable=True, name="fc3")

    ## cost, accuracy, train 지정
    with tf.variable_scope("cost"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=fc3))
        tf.summary.scalar("cross_entropy", cross_entropy)

    with tf.variable_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(fc3, 1), tf.argmax(y, 1)), dtype=tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    ema_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.variable_scope("training"):
        with tf.control_dependencies(ema_op):
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    probability = tf.nn.softmax(fc3)

class Alexnet():
    def __init__(self):
        print("아직 미완성입니다.")
