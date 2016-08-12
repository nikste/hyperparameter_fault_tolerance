import numpy as np
from sklearn import svm
import scipy as sp
from sklearn.linear_model import SGDClassifier as SGDC









def train_svm(X, y, X_val=[], y_val=[], gamma='auto', C=0.1, max_iter=1000, kernel='linear'):
    clf = svm.SVC(gamma=gamma, C=C, verbose=False, max_iter=1, kernel=kernel)
    clf = svm.LinearSVC(C=C, verbose=False, max_iter=1)

    errors = []

    for i in range(0, max_iter):
        clf.fit(X, y)
        if X_val.any() and y_val.any():

            if i % 10 == 0:
                # prediction,mean_error = test_svm(clf,X_val,y_val)
                prediction, mean_error = test_svm(clf, X, y)
                errors.append(mean_error)
                print "iteration:", i, "error:", mean_error
    return clf, errors


def test_svm(clf, X_test, y_test):
    prediction = clf.predict(X_test)
    diff = sp.sign(prediction - y_test)
    # print diff
    mean_error = sp.mean(abs(diff))

    return prediction, mean_error


def do_training(clf,X,y,X_val=[],y_val=[],max_iter=1000,alpha=0.0001,verbose=0,termination_threshold=0.001,seed=0):
    errors = []
    clf.partial_fit(X, y, classes=[-1, 1])

    num_iterations = 0
    for i in range(0, max_iter):
        p_old = clf.coef_.copy()
        clf.partial_fit(X, y, classes=[-1, 1])
        p_new = clf.coef_.copy()

        num_iterations += 1
        # print "abs:",np.substract(np.asarray(p_old), np.asarray(p_new))
        w_diff = np.sum(np.absolute(p_old - p_new))

        if i % 1000 == 0:
            # print "    iters:",i,"difference:",np.sum(np.absolute(p_old - p_new))
            prediciton, mean_error = test_svm(clf, X, y)
            errors.append(mean_error)
            print "    iterations:",i,"error:",mean_error

        if w_diff < termination_threshold:
            print "    termination reached: alpha:", alpha, "w_diff:", w_diff, termination_threshold, "    iterations:", i, "error:", mean_error
            prediciton, mean_error = test_svm(clf, X, y)
            errors.append(mean_error)
            return clf, errors, num_iterations
    return clf, errors, num_iterations

def train_sgd(X,y,X_val=[],y_val=[],max_iter=1000, alpha=0.0001,verbose=0,termination_threshold=0.001,seed=0):
    # sp.random.seed(seed)
    clf = SGDC(warm_start=True,alpha=alpha,verbose=verbose,n_iter=1,random_state=seed)
    clf,errors,num_iterations = do_training(clf,X,y,X_val=X_val,y_val=y_val,max_iter=max_iter,alpha=alpha,verbose=verbose,termination_threshold=termination_threshold,seed=seed)
    return clf,errors,num_iterations