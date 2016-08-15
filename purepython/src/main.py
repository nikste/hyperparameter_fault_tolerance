from models import tf_linear_regression
from models import tf_logistic_regression
from experiments import train, train_parallel, warmstart_all_parallel
from data_generator import generate_noisy_linear_data
from data_generator import generate_noisy_polinomial_data
from data_generator import get_mnist
from data_generator import get_iris
from experiments import warmstart,warmstart_parallel,train_all_parallel
from visualizer import visualize_regression_points, visualize_warmstart_result, visualize_training_result, \
    visualize_weight_difference, visualize_warmstart_result_from_parallel, visualize_training_result_from_parallel

# train(dataset='mnist', fname='results_softmax_regression_mnist')

# warmstart(fname_in='results_softmax_regression_iris', dataset='iris', fname_out='results_softmax_regression_warmstart_iris')
# warmstart_parallel(fname_in='/home/nikste/workspace-python/parallel-failure-recovery/experiments_results/mnist/results_softmax_regression_mnist', dataset='mnist', fname_out='results_softmax_regression_warmstart_mnist')

# visualize_warmstart_result("results_softmax_regression_warmstart_mnist")
# visualize_training_result('/home/nikste/workspace-python/parallel-failure-recovery/experiments_results/mnist/results_softmax_regression_mnist')
# visualize_weight_difference('/home/nikste/workspace-python/parallel-failure-recovery/experiments_results/mnist/results_softmax_regression_mnist')




train_all_parallel(dataset='mnist', fname='results_softmax_regression_mnist', model_type='softmax_regression')
warmstart_all_parallel(fname_in='results_softmax_regression_mnist', dataset='mnist', fname_out='results_softmax_regression_warmstart_mnist2', model_type='softmax_regression')
# # warmstart_all_parallel(fname_in='/home/nikste/workspace-python/parallel-failure-recovery/experiments_results/mnist/results_softmax_regression_mnist', dataset='mnist', fname_out='results_softmax_regression_warmstart_mnist')
# visualize_training_result_from_parallel('results_softmax_regression_mnist')
# visualize_warmstart_result_from_parallel("results_softmax_regression_warmstart_mnist")
print "great success!"
# train_all_parallel(dataset='diabetes', fname='results_linear_regression_diabetes', model_type='linear_regression')
# warmstart_all_parallel(fname_in='results_linear_regression_diabetes', model_type='linear_regression', dataset='diabetes', fname_out='results_linear_regression_warmstart_diabetes')
# # warmstart_all_parallel(fname_in='/home/nikste/workspace-python/parallel-failure-recovery/experiments_results/mnist/results_softmax_regression_mnist', dataset='mnist', fname_out='results_softmax_regression_warmstart_mnist')
# visualize_training_result_from_parallel('results_linear_regression_diabetes')
# visualize_warmstart_result_from_parallel("results_linear_regression_warmstart_diabetes")