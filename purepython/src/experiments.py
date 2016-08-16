from sklearn.externals.joblib import delayed

from models import tf_logistic_regression
from data_generator import get_mnist, get_diabetes
from data_generator import get_iris
from models import tf_softmax_regression
from sklearn.externals import joblib

from models import tf_linear_regression

import pickle

def train(dataset='mnist', fname='results_softmax_regression_mnist'):
    train_test_ratio = 0.5
    if dataset == 'mnist':
        x, y, x_test, y_test = get_mnist(train_test_ratio)
    elif dataset == 'iris':
        x, y, x_test, y_test = get_iris(train_test_ratio)

    regularizations = list(reversed([100., 10., 1., 0.1, 0.01, 0.001, 0.0001]))

    results = []
    for reg_i in xrange(0, len(regularizations)):
        intermediate_results = []
        print "reg:", reg_i
        for i in xrange(0, 10):
            print "    i:", i
            reg = regularizations[reg_i]
            r = tf_softmax_regression.train_softmax(x, y, x_test, y_test, learning_rate=0.005,
                                                            max_iterations=1000000, regularization=reg,
                                                            w_diff_term_crit=0.0001, verbose=True)
            intermediate_results.append(r)
        results.append(intermediate_results)

    # print results
    for el_i in xrange(len(results)):
        print el_i
        el = results[el_i]
        for e in el:
            print e
    import pickle
    pickle.dump(results, open(fname, 'wb'))


def train_all_parallel(x, y, x_test, y_test, fname='results_softmax_regression_mnist', model_type='softmax_regression',  w_diff_term_crit=0.0001, learning_rate=0.0001, regularizations = [100., 10., 1., 0.1, 0.01, 0.001, 0.]):
    if model_type == 'softmax_regression':
        results = joblib.Parallel(n_jobs=47)(delayed( tf_softmax_regression.train_softmax)(
            x, y, x_test, y_test, learning_rate=learning_rate, max_iterations=1000000,
            regularization=regularizations[reg_i], w_diff_term_crit=w_diff_term_crit, verbose=True) for i_par in range(10) for reg_i in xrange(0, len(regularizations)))

    elif model_type == 'linear_regression':
        results = joblib.Parallel(n_jobs=47)(delayed(tf_linear_regression.train)(
            x, y, x_test, y_test, learning_rate=learning_rate, max_iterations=1000000,
            regularization=regularizations[reg_i], w_diff_term_crit=w_diff_term_crit, verbose=True) for i_par in range(10) for
                                             reg_i in xrange(0, len(regularizations)))

    pickle.dump(results, open(fname, 'wb'))


def warmstart_all_parallel(x, y, x_test, y_test, fname_in='results_softmax_regression_mnist', fname_out='results_softmax_regression_warmstart_mnist', model_type='softmax_regression', w_diff_term_crit=0.0001, learning_rate=0.0001, regularizations = [100., 10., 1., 0.1, 0.01, 0.001, 0.]):
    pretrained_models = pickle.load(open(fname_in, 'rb'))
    if model_type == 'softmax_regression':
        #previous_loss_train=None, previous_regularization_penalty_train=None
        results = joblib.Parallel(n_jobs=47)(delayed(tf_softmax_regression.train_softmax)
                                             (
                                             x, y, x_test, y_test, learning_rate=learning_rate, max_iterations=1000000,
                                             w_diff_term_crit=w_diff_term_crit, verbose=True,
                                             regularization=regularizations[target_i],
                                             model=pretrained_models[init_i]['model'],
                                             regularization_initialization=pretrained_models[init_i]['regularization'],
                                             previous_loss_train=pretrained_models[init_i]['loss_train'],
                                             previous_regularization_penalty_train=pretrained_models[init_i]['regularization_penalty_train']
                                         ) for target_i in xrange(0, len(regularizations))
                                           for init_i in xrange(0, len(pretrained_models))
                                         )
    elif model_type == 'linear_regression':
        results = joblib.Parallel(n_jobs=47)(delayed(tf_linear_regression.train)
                                                 (
                                                 x, y, x_test, y_test, learning_rate=learning_rate, max_iterations=1000000,
                                                 w_diff_term_crit=w_diff_term_crit, verbose=True,
                                                 regularization=regularizations[target_i],
                                                 model=pretrained_models[init_i]['model'],
                                                 regularization_initialization=pretrained_models[init_i][
                                                     'regularization']
                                             ) for target_i in xrange(0, len(regularizations))
                                             for init_i in xrange(0, len(pretrained_models))
                                             )
    pickle.dump(results, open(fname_out, 'wb'))


def warmstart(fname_in='results_softmax_regression_mnist', dataset='mnist', fname_out='results_softmax_regression_warmstart_mnist'):
    train_test_ratio = 0.5
    if dataset == 'mnist':
        x, y, x_test, y_test = get_mnist(train_test_ratio)
    elif dataset == 'iris':
        x, y, x_test, y_test = get_iris(train_test_ratio)

    regularizations = [100., 10., 1., 0.1, 0.01, 0.001, 0.0001]

    import pickle
    pretrained_models = pickle.load(open(fname_in, 'rb'))

    results = []
    for reg_i in xrange(0, len(pretrained_models)):
        # all have same regularization parameter
        intermediate_results = []
        print "reg:", reg_i
        for i in xrange(0, len(pretrained_models[reg_i])):
            print "    i:", i
            r_int = []
            reg_warmstart = pretrained_models[reg_i][i]['regularization']
            model = pretrained_models[reg_i][i]['model:']

            for j in xrange(0, len(regularizations)):
                reg = regularizations[j]
                print "----training:", reg, "with initialization of:", reg_warmstart
                # r = tf_linear_regression.train_single_output(x, y, x_test, y_test, learning_rate=0.0001, max_iterations=1000000, regularization=reg, w_diff_term_crit=0.000001, verbose=True)
                r = tf_logistic_regression.train_softmax_output(x, y, x_test, y_test, learning_rate=0.1,
                                                                max_iterations=1000000, regularization=reg,
                                                                w_diff_term_crit=0.000001, verbose=True, model=model)

                r['initialized_with_regularization'] = reg_warmstart
                r_int.append(r)

            intermediate_results.append(r_int)
        results.append(intermediate_results)

    pickle.dump(results, open(fname_out, 'wb'))