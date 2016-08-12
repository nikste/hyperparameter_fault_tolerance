from models import tf_linear_regression
from models import tf_logistic_regression
from experiments import train, train_parallel
from data_generator import generate_noisy_linear_data
from data_generator import generate_noisy_polinomial_data
from data_generator import get_mnist
from data_generator import get_iris
from experiments import warmstart,warmstart_parallel,train_all_parallel
from visualizer import visualize_regression_points, visualize_warmstart_result


# train(dataset='mnist', fname='results_softmax_regression_mnist')
train_all_parallel(dataset='mnist', fname='results_softmax_regression_mnist')

# warmstart(fname_in='results_softmax_regression_iris', dataset='iris', fname_out='results_softmax_regression_warmstart_iris')
# warmstart_parallel(fname_in='results_softmax_regression_mnist', dataset='mnist', fname_out='results_softmax_regression_warmstart_mnist')
# visualize_warmstart_result("results_softmax_regression_warmstart_mnist")