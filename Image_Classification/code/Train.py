## 패키지 import
import os
import sys
import Model
import Load_data
import numpy as np
import tensorflow as tf
sys.path.append(os.getcwd() + "/Image_Classification/code/")

if __name__ == "__main__":
    pre_train_path = os.getcwd() + "/Image_Classification/Pretrain/Basic/"
    train_path = os.getcwd() + "/Image_Classification/Data/train"
    log_path = os.getcwd() + "/Image_Classification/Log"
    train_x, train_y = Load_data.Train_generator(train_path)

    model = Model.DenseNet(number_of_label=2, growth_rate=32)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    merge = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_path, sess.graph)

    ## training 시작
    for i in range(201):
        _, ema_tr = sess.run([model.train_step, model.ema_op], model.feed_dict(True, train_x, train_y, batch_size=5))
        if i % 100 == 0:
            _, ema_tr, acc, loss, summary = sess.run([model.train_step, model.ema_op, model.accuracy,
                                                      model.cross_entropy, merge],
                                                     model.feed_dict(True, train_x, train_y, batch_size=5))
            train_writer.add_summary(summary, i)
            print("step : %d, train accuracy : %g, train loss : %g" % (i, acc, loss))

    tf.train.Saver().save(sess, pre_train_path)