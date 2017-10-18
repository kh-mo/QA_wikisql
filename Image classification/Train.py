## 패키지 import
import os
import sys
sys.path.append(os.getcwd() + "/image classification/")
import Model
import Load_data
import numpy as np
import tensorflow as tf

pre_train_path = os.getcwd() + "/Image classification/Pretrain/Basic/"
train_path = os.getcwd() + "/Image classification/Data/train"
log_path = os.getcwd() + "/Image classification/Log"
train_x, train_y = Load_data.Train_generator(train_path)

def feed_dict(train):
    batch_idx = np.random.choice(train_x.shape[0], 5, False)
    xs, ys = train_x[batch_idx], train_y[batch_idx]
    bool = True
    return {model.x: xs, model.y: ys, model.bol: bool}

model = Model.Basic()
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
merge = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(log_path, sess.graph)

## training 시작
for i in range(201):
    _, ema_tr = sess.run([model.train_step, model.ema_op], feed_dict(True))
    if i%100 == 0:
        _, ema_tr, acc, loss, summary  = sess.run([model.train_step, model.ema_op, model.accuracy, model.cross_entropy, merge], feed_dict(True))
        train_writer.add_summary(summary, i)
        print("step : %d, train accuracy : %g, train loss : %g"%(i, acc, loss))

tf.train.Saver().save(sess, pre_train_path)

