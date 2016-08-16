import datetime
import tensorflow as tf
import numpy as np
import math
import sys


def train_softmax(x, y, x_test, y_test, learning_rate=0.01, max_iterations=1000000, regularization=1., w_diff_term_crit=0.0001, verbose=False, model=None, regularization_initialization=None, previous_loss_train=None, previous_regularization_penalty_train=None):

    print "starting training reg", regularization, "init_reg", regularization_initialization, datetime.datetime.now()
    sys.stdout.flush()
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

    # print x.shape
    # print y.shape
    #
    # print x_test.shape
    # print y_test.shape
    # set up constants
    num_input_dims = x.shape[1]
    num_label_dims = y.shape[1]

    reg_fact = tf.constant(regularization, name='regularization_factor')

    with tf.name_scope('input'):
        x_input = tf.placeholder(tf.float32, shape=[None, num_input_dims], name='input')
    with tf.name_scope('target'):
        y_ = tf.placeholder(tf.float32, shape=[None, num_label_dims], name='target')

    # linear regression
    with tf.name_scope('softmax'):
        # init_vals = , name='truncated_normal_init_val_w')
        if model == None:
            w = tf.Variable(tf.truncated_normal([num_input_dims, num_label_dims], stddev=1. / math.sqrt(2)), name='w')
            # w = tf.Variable(tf.zeros([num_input_dims, num_label_dims]), name='w')
            b = tf.Variable(tf.zeros([num_label_dims]), name='b')
            # b = tf.Variable(tf.zeros([num_label_dims]), name='b')
        else:
            w = tf.Variable(model[0], name='w')
            b = tf.Variable(model[1], name='b')

        output = tf.nn.softmax(tf.matmul(x_input, w) + b)

    with tf.name_scope('regularization'):
        # regularization_penalty = tf.reduce_sum(w, name='regularization_penalty_sum')
        regularization_penalty = (tf.reduce_sum(tf.square(w)) * reg_fact)

    with tf.name_scope('loss'):
        # squared error loss + regularizationPenalty
        log_output = tf.log(output + 1e-9)
        sum_reduction = - tf.reduce_sum(y_ * log_output)
        loss = sum_reduction + regularization_penalty

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.name_scope('optimizer'):
        opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        # grads = opt.compute_gradients(loss)
        # opt = opt.apply_gradients(grads)

    init = tf.initialize_all_variables()
    sess = tf.Session()

    sess.run(init)

    its = 0
    loss_train = -1.

    batch_size = x.shape[0]
    w_old = sess.run(w)
    loss_old = 0.
    # if verbose:
    #     print "w_old",w_old
    t_start = datetime.datetime.now()


    # convergence criteria:
    # 5 consecutive error changes below threshold
    error_changes_past = [100.,100.,100.,100.,100.]

    for i in xrange(0, max_iterations):
        # shuffle input data:
        #per = np.random.permutation(range(0,x.shape[0]))
        # x = x[per]
        # y = y[per]

        for ii in xrange(0, len(x), batch_size):
            log_output__, sum_reduction__, w__, b__,output__, accuracy__, loss__, _, regularization_penalty__ = sess.run([log_output, sum_reduction, w, b, output, accuracy, loss, opt, regularization_penalty], feed_dict={x_input: x[i:i + batch_size,:], y_: y[i:i + batch_size]})
            loss_new = loss__
            # log_output__, sum_reduction__, w__, b__,output__, accuracy__, loss__, regularization_penalty__ = sess.run([log_output, sum_reduction, w, b, output, accuracy, loss, regularization_penalty], feed_dict={x_input: x, y_: y})

        w_new = sess.run(w)
        w_diff = np.sum(np.abs(w_new - w_old))
        loss_diff = np.abs(loss_old - loss_new)
        # if i % 1 == 0:
        #     # print i, "reg", regularization, "init_reg", regularization_initialization, "w_diff:", w_diff, "loss_diff:", loss_diff
        #     # print i, "reg", regularization, "init_reg", regularization_initialization, "\nloss", loss__, "\nregularization_penalty", regularization_penalty__,"\n"#, "w:\n", w__, "b:\n", b__#"w_diff:", w_diff, "loss_diff:", loss_diff
        #     print i, "loss", previous_loss_train, loss__, "regularization_penalty", previous_regularization_penalty_train, regularization_penalty__,"\n"#, "w:\n", w__, "b:\n", b__#"w_diff:", w_diff, "loss_diff:", loss_diff
        w_old = w_new
        loss_old = loss_new

        error_changes_past.append(loss_diff)
        error_changes_past.pop(0)

        # if i % 1000 == 0:
        #     t_end = datetime.datetime.now()
        #     accuracy__ = sess.run([accuracy], feed_dict={x_input: x, y_: y})
        #     print i,"reg", regularization, "init_reg", regularization_initialization, "accuracy:", accuracy__, "sum_red", sum_reduction__ , "reg_penalty", regularization_penalty__, "loss:", loss__, "weight_diff", w_diff
        #     print "took:", t_end - t_start
        #     t_start = t_end
        #     # print "output:", output__
        #     # print "log_output:", log_output__
        #     # print "sum_reduction:", sum_reduction__
        # todo include termination criterion (weight change)
        # if w_diff < w_diff_term_crit and i != 0:
        # if loss_diff < w_diff_term_crit and i != 0:
        if np.sum(loss_diff) < w_diff_term_crit * 5:
            if verbose:
                accuracy__ = sess.run([accuracy], feed_dict={x_input: x, y_: y})
                break
            break

    w_old = sess.run(w)
    accuracy_test = sess.run([accuracy], feed_dict={x_input: x_test, y_: y_test})
    accuracy_train = sess.run([accuracy], feed_dict={x_input: x, y_: y})

    loss_train, regularization_penalty_train = sess.run([loss,regularization_penalty], feed_dict={x_input: x, y_: y})
    loss_test, regularization_penalty_test = sess.run([loss,regularization_penalty], feed_dict={x_input: x_test, y_: y_test})
    print "returning:"
    print "w:\n", w__
    print "b:\n", b__
    res_dict = {"loss_train": loss_train,
                "regularization_penalty_train": regularization_penalty_train,
                "loss_test": loss_test,
                "regularization_penalty_test": regularization_penalty_test,
                "regularization": regularization, "iterations": i, "accuracy_test": accuracy_test, "accuracy_train": accuracy_train, "model": (w__,b__)}

    if regularization_initialization != None:
        res_dict['initialized_with_regularization'] = regularization_initialization
    sess.close()
    tf.reset_default_graph()
    print "finished", i, "reg", \
        regularization, "init_reg", regularization_initialization, \
        "accuracy_train", accuracy_train, "accuracy_test", accuracy_test, \
        "loss_train", loss_train, \
        "regularization_penalty_train", regularization_penalty_train, \
        "loss_test", loss_test, \
        "regularization_penalty_test", regularization_penalty_test, datetime.datetime.now()
    sys.stdout.flush()
    return res_dict