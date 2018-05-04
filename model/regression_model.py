#!/usr/bin/env python

import argparse
import os

import pandas as pd
import tensorflow as tf
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.python.lib.io import file_io


def load_data(file_path):
    file_stream = file_io.FileIO(file_path, mode='r')
    df = pd.read_csv(file_stream)
    return df['x'], df['y']


def model(X, w):
    return tf.multiply(X, w)


def train_model(args):
    trX, trY = load_data(args.trainingFile)
    X = tf.placeholder("float")
    Y = tf.placeholder("float")
    w = tf.Variable(0.0, name="weights")
    y_model = model(X, w)

    cost = tf.square(Y - y_model)

    train_op = tf.train.GradientDescentOptimizer(args.learningRate).minimize(cost)

    eval_path = os.path.join(args.jobDir, 'logs')
    summary_writer = tf.summary.FileWriter(eval_path)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(100):
            for (x, y) in zip(trX, trY):
                sess.run(train_op, feed_dict={X: x, Y: y})

        tf.saved_model.simple_save(sess, os.path.join(args.exportDir,
                                                      "learning_rate="
                                                      + str(args.learningRate)),
                                   inputs={"X": X},
                                   outputs={"Y": Y})
        final_cost = sess.run(tf.reduce_sum(tf.square(Y - y_model)), feed_dict={X: trX, Y: trY})
        summary = Summary(value=[Summary.Value(tag='hyperparameterMetricTag', simple_value=final_cost)])
        summary_writer.add_graph(sess.graph)
        summary_writer.add_summary(summary)
        summary_writer.flush()
        print(sess.run(w))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir",
                        metavar="DIR",
                        action="store",
                        dest="exportDir",
                        default="../saved_models",
                        help="Directory for storing trained model")
    parser.add_argument("--job-dir",
                        metavar="DIR",
                        action="store",
                        dest="jobDir",
                        default="../local-job-dir",
                        help="Directory for the job data")
    parser.add_argument("--learning-rate",
                        metavar="RATE",
                        action="store",
                        dest="learningRate",
                        type=float,
                        default=0.01,
                        help="Learning Rate")
    parser.add_argument("--train-file",
                        metavar="TRAINFILE",
                        action="store",
                        dest="trainingFile",
                        type=str,
                        default="../data/random_linear.csv",
                        help="File path containing the training data .csv file")
    args = parser.parse_args()
    train_model(args)
