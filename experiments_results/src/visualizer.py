import matplotlib.pyplot as plt
import numpy as np
import pickle


def visualize_2d_2class_points(x,y):
    plt.scatter(x[y == 0, 0], x[y == 0, 1], c="red")
    plt.scatter(x[y == 1, 0], x[y == 1, 1], c="blue")
    plt.show()


def visualize_regression_points(x,y):
    plt.scatter(x,y)
    plt.axes().set_aspect('equal','datalim')
    plt.show()


def rewrite_input_data(input_d):
    d_sorted = {}
    # create sorted dictionary for different regularizations
    for el in input_d:
        for elel in el:
            print elel['regularization']
            reg_k = str(elel['regularization'])
            if reg_k not in d_sorted.keys():
                d_sorted[reg_k] = {'iterations':[elel['iterations']],
                                   'accuracy_train':[elel['accuracy_train']],
                                   'accuracy_test':[elel['accuracy_test']],
                                   'model':[elel['model:']]}
            else:
                d_sorted[reg_k]['iterations'].append(elel['iterations'])
                d_sorted[reg_k]['accuracy_train'].append(elel['accuracy_train'])
                d_sorted[reg_k]['accuracy_test'].append(elel['accuracy_test'])
                d_sorted[reg_k]['model'].append(elel['model:'])

    return d_sorted


def rewrite_input_data_from_parallel(input_d):
    d_sorted = {}

    # create sorted dictionary for different regularizations
    for el in input_d:
        print el['regularization']
        reg_k = str(el['regularization'])
        if reg_k not in d_sorted.keys():
            d_sorted[reg_k] = {'iterations':[el['iterations']],
                               'accuracy_train':[el['accuracy_train']],
                               'accuracy_test':[el['accuracy_test']],
                               'model':[el['model']]}
        else:
            d_sorted[reg_k]['iterations'].append(el['iterations'])
            d_sorted[reg_k]['accuracy_train'].append(el['accuracy_train'])
            d_sorted[reg_k]['accuracy_test'].append(el['accuracy_test'])
            d_sorted[reg_k]['model'].append(el['model'])

    return d_sorted


def visualize_weight_difference(filename):
    d = pickle.load(open(filename, 'rb'))

    d_sorted = rewrite_input_data(d)

    print "regularization\tsum_m_model\tsum_var_model"
    # aggregate statistics
    for k in list(sorted(d_sorted.keys())):
        el = d_sorted[k]


        weights = []
        for w in el['model']:
            weights.append(el['model'][0])
        m_model = np.mean(weights)
        var_model = np.std(weights)

        # not sure if this good measure:
        print k, "\t", np.mean(m_model), "\t", np.std(var_model)


def visualize_training_result(filename):
    d = pickle.load(open(filename, 'rb'))

    d_sorted = rewrite_input_data(d)

    print "regularization\tm_iterations\tvar_itartions\tm_accuracy_train\tvar_accuracy_train\tm_accuracy_test\tvar_accuracy_test"
    # aggregate statistics
    for k in list(sorted(d_sorted.keys())):
        el = d_sorted[k]
        m_iterations = np.mean(el['iterations'])
        var_iterations = np.std(el['iterations'])

        m_accuracy_test = np.mean(el['accuracy_test'])
        var_accuracy_test = np.std(el['accuracy_test'])

        m_accuracy_train = np.mean(el['accuracy_train'])
        var_accuracy_train = np.std(el['accuracy_train'])

        print k, "\t", m_iterations, "\t", var_iterations, "\t", \
            m_accuracy_train, "\t", var_accuracy_train, "\t", \
            m_accuracy_test, "\t", var_accuracy_test


def visualize_training_result_from_parallel(filename):
    d = pickle.load(open(filename, 'rb'))

    d_sorted = rewrite_input_data_from_parallel(d)

    print "regularization\tm_iterations\tvar_itartions\tm_accuracy_train\tvar_accuracy_train\tm_accuracy_test\tvar_accuracy_test"
    # aggregate statistics
    for k in list(sorted(d_sorted.keys())):
        el = d_sorted[k]
        m_iterations = np.mean(el['iterations'])
        var_iterations = np.std(el['iterations'])

        m_accuracy_test = np.mean(el['accuracy_test'])
        var_accuracy_test = np.std(el['accuracy_test'])

        m_accuracy_train = np.mean(el['accuracy_train'])
        var_accuracy_train = np.std(el['accuracy_train'])

        print k, "\t", m_iterations, "\t", var_iterations, "\t", \
            m_accuracy_train, "\t", var_accuracy_train, "\t", \
            m_accuracy_test, "\t", var_accuracy_test


def visualize_warmstart_result_from_parallel(filename, filename_init):
    d_init = pickle.load(open(filename_init, 'rb'))

    d_sorted_init = rewrite_input_data_from_parallel(d_init)

    d = pickle.load(open(filename, 'rb'))

    regularizations = []
    d_sorted = {}
    for el in d:
        r = str(el['regularization'])
        r_init = str(el['initialized_with_regularization'])
        if r not in d_sorted.keys():
            d_sorted[r] = {r_init: {'iterations': [el['iterations']],
                                    'accuracy_train': [el['accuracy_train']],
                                    'accuracy_test': [el['accuracy_test']]}}
        elif r_init not in d_sorted[r].keys():
            d_sorted[r][r_init] = {'iterations': [el['iterations']],
                                   'accuracy_train': [el['accuracy_train']],
                                   'accuracy_test': [
                                       el['accuracy_test']]}  # todo: think if this is necessairy
        else:
            d_sorted[r][r_init]['iterations'].append(el['iterations'])
            d_sorted[r][r_init]['accuracy_train'].append(el['accuracy_train'])
            d_sorted[r][r_init]['accuracy_test'].append(el['accuracy_test'])

    # compute mean and variance for each list ['regularization']['initialized_with_regularization']
    k_reg = d_sorted.keys()
    k_reg_init = d_sorted[k_reg[0]].keys()

    print  "reg\t", "reg_init\t", "mean\t", "variance\t", "mean_acc_train|loss_train\t", "var_acc_train|loss_train\t", "acc_test|loss_test\t", "var_acc_test|loss_test", "without warmstart"

    for k_reg in sorted(d_sorted.keys()):

        # reference values
        el = d_sorted_init[k_reg]
        m_iterations_init = np.mean(el['iterations'])

        for k_reg_init in sorted(d_sorted[k_reg].keys()):
            # print d_sorted[k_reg][k_reg_init]['iterations']
            m_iterations = np.mean(d_sorted[k_reg][k_reg_init]['iterations'])
            var_iterations = np.std(d_sorted[k_reg][k_reg_init]['iterations'])
            d_sorted[k_reg][k_reg_init]['mean_iterations'] = m_iterations
            d_sorted[k_reg][k_reg_init]['variance_iterations'] = var_iterations

            m_accuracy_train = np.mean(d_sorted[k_reg][k_reg_init]['accuracy_train'])
            var_accuracy_train = np.std(d_sorted[k_reg][k_reg_init]['accuracy_train'])
            d_sorted[k_reg][k_reg_init]['mean_loss_train'] = m_accuracy_train
            d_sorted[k_reg][k_reg_init]['variance_loss_train'] = var_accuracy_train

            m_accuracy_test = np.mean(d_sorted[k_reg][k_reg_init]['accuracy_test'])
            var_accuracy_test = np.std(d_sorted[k_reg][k_reg_init]['accuracy_test'])
            d_sorted[k_reg][k_reg_init]['mean_accuracy_test'] = m_accuracy_test
            d_sorted[k_reg][k_reg_init]['variance_accuracy_test'] = var_accuracy_test

            print k_reg, "\t", k_reg_init, "\t", m_iterations, "\t", var_iterations, "\t", \
                m_accuracy_train, "\t", var_accuracy_train, "\t", \
                m_accuracy_test, "\t", var_accuracy_test, "\t", m_iterations_init
        print " "


def visualize_warmstart_result(filename):
    d = pickle.load(open(filename,'rb'))

    d_sorted = {}
    for el in d:
        for ell in el:
            for elll in ell:
                r = str(elll['regularization'])
                r_init = str(elll['initialized_with_regularization'])
                if r not in d_sorted.keys():
                    d_sorted[r] = {r_init: {'iterations':[elll['iterations']],
                                            'accuracy_train':[elll['accuracy_train']],
                                            'accuracy_test':[elll['accuracy_test']]}}
                elif r_init not in d_sorted[r].keys():
                    d_sorted[r][r_init] = {'iterations': [elll['iterations']],
                                            'accuracy_train':[elll['accuracy_train']],
                                            'accuracy_test':[elll['accuracy_test']]} # todo: think if this is necessairy
                else:
                    d_sorted[r][r_init]['iterations'].append(elll['iterations'])
                    d_sorted[r][r_init]['accuracy_train'].append(elll['accuracy_train'])
                    d_sorted[r][r_init]['accuracy_test'].append(elll['accuracy_test'])




    # compute mean and variance for each list ['regularization']['initialized_with_regularization']
    k_reg = d_sorted.keys()
    k_reg_init = d_sorted[k_reg[0]].keys()

    print  "reg\t", "reg_init\t", "mean\t", "variance\t", "mean_acc_train|loss_train\t", "var_acc_train|loss_train\t", "acc_test|loss_test\t", "var_acc_test|loss_test"

    for k_reg in sorted(d_sorted.keys()):
        for k_reg_init in sorted(d_sorted[k_reg].keys()):
            #print d_sorted[k_reg][k_reg_init]['iterations']
            m_iterations = np.mean(d_sorted[k_reg][k_reg_init]['iterations'])
            var_iterations = np.std(d_sorted[k_reg][k_reg_init]['iterations'])
            d_sorted[k_reg][k_reg_init]['mean_iterations'] = m_iterations
            d_sorted[k_reg][k_reg_init]['variance_iterations'] = var_iterations

            m_accuracy_train = np.mean(d_sorted[k_reg][k_reg_init]['accuracy_train'])
            var_accuracy_train = np.std(d_sorted[k_reg][k_reg_init]['accuracy_train'])
            d_sorted[k_reg][k_reg_init]['mean_loss_train'] = m_accuracy_train
            d_sorted[k_reg][k_reg_init]['variance_loss_train'] = var_accuracy_train

            m_accuracy_test = np.mean(d_sorted[k_reg][k_reg_init]['accuracy_test'])
            var_accuracy_test = np.std(d_sorted[k_reg][k_reg_init]['accuracy_test'])
            d_sorted[k_reg][k_reg_init]['mean_accuracy_test'] = m_accuracy_test
            d_sorted[k_reg][k_reg_init]['variance_accuracy_test'] = var_accuracy_test

            print k_reg,"\t", k_reg_init,"\t", m_iterations,"\t", var_iterations, "\t", \
                m_accuracy_train, "\t", var_accuracy_train, "\t", \
                m_accuracy_test, "\t", var_accuracy_test
        print " "
