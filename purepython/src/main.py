from models import tf_linear_regression
from models import tf_logistic_regression
from experiments import train, warmstart_all_parallel
from data_generator import generate_noisy_linear_data, get_diabetes
from data_generator import generate_noisy_polinomial_data
from data_generator import get_mnist
from data_generator import get_iris
from experiments import warmstart,train_all_parallel
from visualizer import visualize_regression_points, visualize_warmstart_result, visualize_training_result, \
    visualize_weight_difference, visualize_warmstart_result_from_parallel, visualize_training_result_from_parallel

# train(dataset='mnist', fname='results_softmax_regression_mnist')

# warmstart(fname_in='results_softmax_regression_iris', dataset='iris', fname_out='results_softmax_regression_warmstart_iris')
# warmstart_parallel(fname_in='/home/nikste/workspace-python/parallel-failure-recovery/experiments_results/mnist/results_softmax_regression_mnist', dataset='mnist', fname_out='results_softmax_regression_warmstart_mnist')

# visualize_warmstart_result("results_softmax_regression_warmstart_mnist")
# visualize_training_result('/home/nikste/workspace-python/parallel-failure-recovery/experiments_results/mnist/results_softmax_regression_mnist')
# visualize_weight_difference('/home/nikste/workspace-python/parallel-failure-recovery/experiments_results/mnist/results_softmax_regression_mnist')

def one_run(dataset, modeltype):
    train_test_ratio = 0.5
    if dataset == 'mnist':
        x, y, x_test, y_test = get_mnist(train_test_ratio)
    elif dataset == 'iris':
        x, y, x_test, y_test = get_iris(train_test_ratio)
    elif dataset == 'diabetes':
        x, y, x_test, y_test = get_diabetes(train_test_ratio)

    f_name_train = 'results_' + model_type + '_' + dataset
    f_name_warmstart = 'results_warmstart_' + model_type + '_' + dataset
    train_all_parallel(x, y, x_test, y_test,fname=f_name_train, model_type=model_type, w_diff_term_crit=w_diff_term_crit, learning_rate=learning_rate, regularizations=regularizations)
    warmstart_all_parallel(x, y, x_test, y_test, fname_in=f_name_train, fname_out=f_name_warmstart, model_type=model_type, w_diff_term_crit=w_diff_term_crit, learning_rate=learning_rate, regularizations=regularizations)
    # # warmstart_all_parallel(fname_in='/home/nikste/workspace-python/parallel-failure-recovery/experiments_results/mnist/results_softmax_regression_mnist', dataset='mnist', fname_out='results_softmax_regression_warmstart_mnist')
    visualize_training_result_from_parallel(f_name_train)
    visualize_warmstart_result_from_parallel(f_name_warmstart)
    print "great success!"
# train_all_parallel(dataset='diabetes', fname='results_linear_regression_diabetes', model_type='linear_regression')
# warmstart_all_parallel(fname_in='results_linear_regression_diabetes', model_type='linear_regression', dataset='diabetes', fname_out='results_linear_regression_warmstart_diabetes')
# # warmstart_all_parallel(fname_in='/home/nikste/workspace-python/parallel-failure-recovery/experiments_results/mnist/results_softmax_regression_mnist', dataset='mnist', fname_out='results_softmax_regression_warmstart_mnist')
# visualize_training_result_from_parallel('results_linear_regression_diabetes')
# visualize_warmstart_result_from_parallel("results_linear_regression_warmstart_diabetes")

w_diff_term_crit = 0.01
learning_rate = 0.0001
regularizations = list(reversed([100., 10., 1., 0.1, 0.01, 0.001, 0.]))

dataset = 'mnist'
model_type = 'softmax_regression'
one_run(dataset, model_type)

# dataset = 'iris'
# model_type = 'softmax_regression'
# one_run(dataset, model_type)
#
# dataset = 'diabetes'
# model_type = 'linear_regression'
# one_run(dataset, model_type)
# print 'fully done!'