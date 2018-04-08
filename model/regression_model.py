#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import pandas as pd

def load_data():
    df = pd.read_csv('../data/random_linear.csv')
    return df['x'], df['y']


def model(X, w):
    return tf.multiply(X, w) # lr is just X*w so this model line is pretty simple

def train_model():
    trX, trY = load_data()
    X = tf.placeholder("float")  # create symbolic variables
    Y = tf.placeholder("float")
    w = tf.Variable(0.0, name="weights") # create a shared variable (like theano.shared) for the weight matrix
    y_model = model(X, w)

    cost = tf.square(Y - y_model) # use square error for cost function

    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost) # construct an optimizer to minimize cost and fit line to my data

    # Launch the graph in a session
    with tf.Session() as sess:
        # you need to initialize variables (in this case just variable W)
        tf.global_variables_initializer().run()

        for i in range(100):
            for (x, y) in zip(trX, trY):
                sess.run(train_op, feed_dict={X: x, Y: y})
        #TODO: Save model for later deployment

        print(sess.run(w))  # It should be something around 2

if __name__ == "__main__":
    train_model()
