## 패키지 import
import os
import sys
sys.path.append(os.getcwd() + "/image classification/")
import Model
import Load_data
import numpy as np
import tensorflow as tf

pre_train_path = os.getcwd() + "/Image classification/Pretrain/Basic/"
test_path = os.getcwd() + "/Image classification/Data/test"
test_x = Load_data.Test_generator(test_path)

model = Model.Basic()
sess = tf.InteractiveSession()
tf.train.Saver().restore(sess, pre_train_path)

prob = sess.run(model.probability, {model.x: test_x, model.bol: False})
print("남성일 확률 : %g, 여성일 확률 : %g"%(prob[0][0] , prob[0][1]))

sess.run(model.probability, {model.x: test_x, model.bol: bool})
sess.run(model.fc3, {model.x: test_x, model.bol: bool})

