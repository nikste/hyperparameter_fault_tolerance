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


def visualize_training_result(filename):
    d = pickle.load(open(filename,'rb'))

    d_sorted = {}
    for el in d:
        for ell in el:
            for elll in ell:
                r = str(elll['regularization'])
                r_init = str(elll['initialized_with_regularization'])
                if r not in d_sorted.keys():
                    d_sorted[r] = {r_init: {'iterations': [elll['iterations']],
                                            'loss_train': [elll['loss_train']],
                                            'loss_test': [elll['loss_test']]}}
                elif r_init not in d_sorted[r].keys():
                    d_sorted[r][r_init] = {'iterations': [elll['iterations']],
                                           'loss_train': [elll['loss_train']],
                                           'loss_test': [elll['loss_test']]}  # todo: think if this is necessairy
                else:
                    d_sorted[r][r_init]['iterations'].append(elll['iterations'])
                    d_sorted[r][r_init]['loss_train'].append(elll['loss_train'])
                    d_sorted[r][r_init]['loss_test'].append(elll['loss_test'])

    # compute mean and variance for each list ['regularization']['initialized_with_regularization']
    k_reg = d_sorted.keys()
    k_reg_init = d_sorted[k_reg[0]].keys()

    print "reg\t", "reg_init\t", "mean\t", "variance\t", "mean_acc_train\t", "var_acc_train\t", "acc_test\t", "var_acc_test"

    for k_reg in sorted(d_sorted.keys()):
        for k_reg_init in sorted(d_sorted[k_reg].keys()):
            # print d_sorted[k_reg][k_reg_init]['iterations']
            m_iterations = np.mean(d_sorted[k_reg][k_reg_init]['iterations'])
            var_iterations = np.std(d_sorted[k_reg][k_reg_init]['iterations'])
            d_sorted[k_reg][k_reg_init]['mean_iterations'] = m_iterations
            d_sorted[k_reg][k_reg_init]['variance_iterations'] = var_iterations

            m_accuracy_test = np.mean(d_sorted[k_reg][k_reg_init]['accuracy_test'])
            var_accuracy_test = np.std(d_sorted[k_reg][k_reg_init]['accuracy_test'])
            d_sorted[k_reg][k_reg_init]['mean_accuracy_test'] = m_accuracy_test
            d_sorted[k_reg][k_reg_init]['variance_accuracy_test'] = var_accuracy_test

            m_accuracy_train = np.mean(d_sorted[k_reg][k_reg_init]['accuracy_train'])
            var_accuracy_train = np.std(d_sorted[k_reg][k_reg_init]['accuracy_train'])
            d_sorted[k_reg][k_reg_init]['mean_accuracy_train'] = m_accuracy_train
            d_sorted[k_reg][k_reg_init]['variance_accuracy_test'] = var_accuracy_train

            print k_reg, "\t", k_reg_init, "\t", m_iterations, "\t", var_iterations, "\t", \
                m_accuracy_train, "\t", var_accuracy_train, "\t", \
                m_accuracy_test, "\t", var_accuracy_test

        print " "


def visualize_warmstart_result(filename):
    d = pickle.load(open(filename,'rb'))

    # need: dict['model_params''
    # is: [initialized_with][regularization]

    # get: [regularization][initialized_with][trial]

    regularizations = []
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

    print  "reg\t", "reg_init\t", "mean\t", "variance\t", "mean_acc_train\t", "var_acc_train\t", "acc_test\t", "var_acc_test"

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


    # print d[0][0][0].keys()
    # c = 0
    # for el in d:
    #     print c
    #     for ell in el:
    #         for elll in ell:
    #             print elll['initialized_with_regularization'], elll['regularization'], elll['iterations']
    #             pass
    #     c += 1