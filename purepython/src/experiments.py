from sklearn.externals.joblib import delayed

from models import tf_logistic_regression
from data_generator import get_mnist
from data_generator import get_iris
from models import tf_softmax_regression
from sklearn.externals import joblib

def train(dataset='mnist', fname='results_softmax_regression_mnist'):

    train_test_ratio = 0.5
    if dataset == 'mnist':
        x, y, x_test, y_test = get_mnist(train_test_ratio)
    elif dataset == 'iris':
        x, y, x_test, y_test = get_iris(train_test_ratio)

    # regularizations = [100., 10., 1., 0.1, 0.01, 0.001, 0.0001]
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

def train_parallel(dataset='mnist', fname='results_softmax_regression_mnist'):
    train_test_ratio = 0.5
    if dataset == 'mnist':
        x, y, x_test, y_test = get_mnist(train_test_ratio)
    elif dataset == 'iris':
        x, y, x_test, y_test = get_iris(train_test_ratio)

    regularizations = list(reversed([100., 10., 1., 0.1, 0.01, 0.001, 0.]))#, 0.0001]

    results = []
    for reg_i in xrange(0, len(regularizations)):
        intermediate_results = []
        print "reg:", reg_i

        reg = regularizations[reg_i]

        intermediate_results = joblib.Parallel(n_jobs=10)(delayed( tf_softmax_regression.train_softmax)(
            x, y, x_test, y_test, learning_rate=0.005, max_iterations=1000000,
            regularization=reg, w_diff_term_crit=0.001, verbose=True) for i_par in range(10))
        results.append(intermediate_results)

    # print results
    for el_i in xrange(len(results)):
        print el_i
        el = results[el_i]
        for e in el:
            print e
    import pickle
    pickle.dump(results, open(fname, 'wb'))


def train_all_parallel(dataset='mnist', fname='results_softmax_regression_mnist'):
    train_test_ratio = 0.5
    if dataset == 'mnist':
        x, y, x_test, y_test = get_mnist(train_test_ratio)
    elif dataset == 'iris':
        x, y, x_test, y_test = get_iris(train_test_ratio)

    regularizations = list(reversed([100., 10., 1., 0.1, 0.01, 0.001, 0.]))#, 0.0001]


    results = joblib.Parallel(n_jobs=58)(delayed( tf_softmax_regression.train_softmax)(
        x, y, x_test, y_test, learning_rate=0.005, max_iterations=1000000,
        regularization=regularizations[reg_i], w_diff_term_crit=0.001, verbose=True) for i_par in range(2) for reg_i in xrange(0, len(regularizations)))

    # print results
    for el_i in xrange(len(results)):
        print el_i

    import pickle
    pickle.dump(results, open(fname, 'wb'))


def warmstart_parallel(fname_in='results_softmax_regression_mnist', dataset='mnist', fname_out='results_softmax_regression_warmstart_mnist'):
    '''
    trains models with warmstarting
    :param x:
    :param y:
    :param x_test:
    :param y_test:
    :param models_file:d
    :return:
    '''
    train_test_ratio = 0.5
    if dataset == 'mnist':
        x, y, x_test, y_test = get_mnist(train_test_ratio)
    elif dataset == 'iris':
        x, y, x_test, y_test = get_iris(train_test_ratio)

    regularizations = list(reversed([100., 10., 1., 0.1, 0.01, 0.001, 0.]))

    import pickle
    pretrained_models = pickle.load(open(fname_in, 'rb'))

    results = []
    for reg_i in xrange(0, len(pretrained_models)):
        # all have same regularization parameter
        intermediate_results = []
        print "reg:", reg_i
        for i in xrange(0, len(pretrained_models[reg_i])):
            print "    i:", i
            reg_warmstart = pretrained_models[reg_i][i]['regularization']
            model = pretrained_models[reg_i][i]['model:']
            print "----training: with initialization of:", reg_warmstart
            ii_res = joblib.Parallel(n_jobs=10)(delayed(tf_softmax_regression.train_softmax)(
                x, y, x_test, y_test, learning_rate=0.005, max_iterations=1000000,
                regularization=regularizations[j], w_diff_term_crit=0.001, verbose=True, model=model, regularization_initialization=reg_warmstart) for j in range(0,len(regularizations)))

            intermediate_results.append(ii_res)
        results.append(intermediate_results)

    pickle.dump(results, open(fname_out, 'wb'))

def warmstart(fname_in='results_softmax_regression_mnist', dataset='mnist', fname_out='results_softmax_regression_warmstart_mnist'):
    '''
    trains models with warmstarting
    :param x:
    :param y:
    :param x_test:
    :param y_test:
    :param models_file:d
    :return:
    '''
    # mu = 0.
    # sigma = 10.
    # n_samples = 100
    # w = 1.
    # b = 0.
    # x, y = generate_noisy_linear_data(mu,sigma,n_samples, w, b)
    # x_test, y_test = generate_noisy_linear_data(mu,sigma,n_samples, w, b)

    train_test_ratio = 0.5
    if dataset == 'mnist':
        x, y, x_test, y_test = get_mnist(train_test_ratio)
    elif dataset == 'iris':
        x, y, x_test, y_test = get_iris(train_test_ratio)
    # x, y = generate_noisy_polinomial_data(mu, sigma, n_samples, .1, .1, .1)
    # x_test, y_test = generate_noisy_polinomial_data(mu, sigma, n_samples, .1, .1, .1)
    # import pickle
    #
    # pickle.dump({"x_train": x, "y_train": y, "x_test": x_test, "y_test": y_test, "data_dist":{"mu": mu, "sigma": sigma, "w": w, "b": b, "n_samples": n_samples}}, open('./random_data/random_linear_test_data_with_noise_1000', 'wb'))
    #
    # visualize_regression_points(x, y)

    regularizations = [100., 10., 1., 0.1, 0.01, 0.001, 0.0001]
    # regularizations = [0.001, 0.0001]



    import pickle
    pretrained_models = pickle.load(open(fname_in, 'rb'))

    results = []
    for reg_i in xrange(0, len(pretrained_models)):
        # all have same regularization parameter
        intermediate_results = []
        # print "reg_i", reg_i, "value:", regularizations[reg_i]
        # print "resultslist:", results
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