import tensorflow as tf
import numpy as np
import math

def train_single_output(x, y, x_test, y_test, learning_rate=0.005, max_iterations=1000000, regularization=1., w_diff_term_crit=0.0001, verbose=False, model=None):
    assert(x.shape[1] == x_test.shape[1],
           "train shape:" + str(x.shape) +
           " and test shape:" + str(x_test.shape) +
           " do not match in dimensionality")

    assert(x.shape[0] == y.shape[0],
           "number of training samples:" + str(x.shape) +
           " and number of labels:" + str(y.shape) +
           " do not match!")
    assert(x_test.shape[0] == y_test.shape[0],
           "number of testing samples:" + str(x_test.shape) +
           " and number of labels:" + str(y_test.shape) +
           " do not match!")

    # set up constants
    num_input_dims = x.shape[1]
    num_label_dims = y.shape[1]

    reg_fact = tf.constant(regularization, name='regularization_factor')

    with tf.name_scope('input'):
        x_input = tf.placeholder(tf.float32, shape=[None, num_input_dims], name='input')
    with tf.name_scope('target'):
        y_ = tf.placeholder(tf.float32, shape=[None, num_label_dims], name='target')

    # linear regression
    with tf.name_scope('linear_regression'):
        # init_vals = , name='truncated_normal_init_val_w')
        if model == None:
            w = tf.Variable(tf.truncated_normal([num_input_dims, num_label_dims], stddev=1. / math.sqrt(2)), name='w')
            b = tf.Variable(tf.zeros([num_label_dims]), name='b')
        else:
            w = tf.Variable(model[0], name='w')
            b = tf.Variable(model[1], name='b')
        output = tf.sigmoid(tf.matmul(x_input, w) + b)

    with tf.name_scope('regularization'):
        # regularization_penalty = tf.reduce_sum(w, name='regularization_penalty_sum')
        regularization_penalty = (tf.reduce_sum(tf.square(w)) * reg_fact)

    with tf.name_scope('loss'):
        # squared error loss + regularizationPenalty
        diff = y_ - output
        sq_diff = tf.square(diff)
        loss = tf.reduce_mean(sq_diff) + regularization_penalty
        # loss = tf.reduce_mean(sq_diff)

    with tf.name_scope('optimizer'):
        opt = tf.train.GradientDescentOptimizer(learning_rate)
        grads = opt.compute_gradients(loss)
        opt = opt.apply_gradients(grads)

    init = tf.initialize_all_variables()
    sess = tf.Session()

    sess.run(init)

    its = 0
    loss_train = -1.

    w_old = sess.run(w)[0][0]
    if verbose:
        print "w_old",w_old
    for i in xrange(0,max_iterations):
        w__, b__,output__, diff__, sq_diff__, loss__, _, regularization_penalty__ = sess.run([w, b, output, diff, sq_diff, loss, opt, regularization_penalty], feed_dict={x_input: x, y_: y})
        w_new = sess.run(w)[0][0]
        its += 1
        w_diff = np.sum(np.abs(w_new - w_old))
        # if i % 1000 == 0:
        #     print "regularization_penalty:", regularization_penalty__
        #     print "iteration:", i
        #     print "weight:", w__
        #     print "w_diff:", w_diff
        #     print "w_diff_term_crit:", w_diff_term_crit
        #     print "loss:", loss__

        # todo include termination criterion (weight change)
        if w_diff < w_diff_term_crit:
            if verbose:
                print "regularization", regularization
                print "finished at iteration:", its
                print "weights:", w_new
                print ""
            # print "weight_difference:", w_diff
            break
        w_old = w_new

    loss_test = sess.run([loss], feed_dict={x_input: x_test, y_: y_test})
    loss_train = sess.run([loss], feed_dict={x_input: x, y_: y})
    res_dict = {"regularization": regularization, "iterations": its, "loss_test": loss_test, "loss_train": loss_train, "model": (w__,b__)}
    sess.close()
    tf.reset_default_graph()
    return res_dict


def train_softmax_output(x, y, x_test, y_test, learning_rate=0.005, max_iterations=1000000, regularization=1., w_diff_term_crit=0.0001, verbose=False, model=None):
    assert(x.shape[1] == x_test.shape[1],
           "train shape:" + str(x.shape) +
           " and test shape:" + str(x_test.shape) +
           " do not match in dimensionality")

    assert(x.shape[0] == y.shape[0],
           "number of training samples:" + str(x.shape) +
           " and number of labels:" + str(y.shape) +
           " do not match!")
    assert(x_test.shape[0] == y_test.shape[0],
           "number of testing samples:" + str(x_test.shape) +
           " and number of labels:" + str(y_test.shape) +
           " do not match!")

    # set up constants
    num_input_dims = x.shape[1]
    num_label_dims = y.shape[1]

    reg_fact = tf.constant(regularization, name='regularization_factor')

    with tf.name_scope('input'):
        x_input = tf.placeholder(tf.float32, shape=[None, num_input_dims], name='input')
    with tf.name_scope('target'):
        y_ = tf.placeholder(tf.float32, shape=[None, num_label_dims], name='target')

    # linear regression
    with tf.name_scope('logistic_regression'):
        # init_vals = , name='truncated_normal_init_val_w')
        if model == None:
            w = tf.Variable(tf.truncated_normal([num_input_dims, num_label_dims], stddev=1. / math.sqrt(2)), name='w')
            b = tf.Variable(tf.zeros([num_label_dims]), name='b')
        else:
            w = tf.Variable(model[0], name='w')
            b = tf.Variable(model[1], name='b')

        output = tf.nn.softmax(tf.sigmoid(tf.matmul(x_input, w) + b))

    with tf.name_scope('regularization'):
        # regularization_penalty = tf.reduce_sum(w, name='regularization_penalty_sum')
        regularization_penalty = (tf.reduce_sum(tf.square(w)) * reg_fact)

    with tf.name_scope('loss'):
        # squared error loss + regularizationPenalty
        diff = y_ - output
        sq_diff = tf.square(diff)
        # loss = tf.reduce_mean(sq_diff) + regularization_penalty
        # loss = tf.reduce_mean(sq_diff)
        loss = -tf.reduce_sum(y_ * tf.log(output)) + regularization_penalty

    with tf.name_scope('optimizer'):
        opt = tf.train.GradientDescentOptimizer(learning_rate)
        grads = opt.compute_gradients(loss)
        opt = opt.apply_gradients(grads)

    init = tf.initialize_all_variables()
    sess = tf.Session()

    sess.run(init)

    its = 0
    loss_train = -1.

    w_old = sess.run(w)[0][0]
    if verbose:
        print "w_old",w_old
    for i in xrange(0,max_iterations):
        w__, b__,output__, diff__, sq_diff__, loss__, _, regularization_penalty__ = sess.run([w, b, output, diff, sq_diff, loss, opt, regularization_penalty], feed_dict={x_input: x, y_: y})
        w_new = sess.run(w)[0][0]
        its += 1
        w_diff = np.sum(np.abs(w_new - w_old))
        # if i % 1000 == 0:
        #     print "regularization_penalty:", regularization_penalty__
        #     print "iteration:", i
        #     print "weight:", w__
        #     print "w_diff:", w_diff
        #     print "w_diff_term_crit:", w_diff_term_crit
        #     print "loss:", loss__

        # todo include termination criterion (weight change)
        if w_diff < w_diff_term_crit:
            if verbose:
                print "regularization", regularization
                print "finished at iteration:", its
                print "weights:", w_new
                print ""
            # print "weight_difference:", w_diff
            break
        w_old = w_new

    loss_test = sess.run([loss], feed_dict={x_input: x_test, y_: y_test})
    loss_train = sess.run([loss], feed_dict={x_input: x, y_: y})
    res_dict = {"regularization": regularization, "iterations": its, "loss_test": loss_test, "loss_train": loss_train, "model:": (w__,b__)}
    sess.close()
    tf.reset_default_graph()
    return res_dict
