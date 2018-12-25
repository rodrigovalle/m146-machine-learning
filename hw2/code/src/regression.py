# This code was adapted from course material by Jenna Wiens (UMichigan).

# python libraries
import os

# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt
from scipy.interpolate import spline

# unit tests
import unittest
import timeit
from functools import partial

######################################################################
# classes
######################################################################

class Data:

    def __init__(self, X=None, y=None):
        """
        Data class.

        Attributes
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
        """

        # n = number of examples, d = dimensionality
        self.X = X
        self.y = y

    def load(self, filename):
        """
        Load csv file into X array of features and y array of labels.

        Parameters
        --------------------
            filename -- string, filename
        """

        # determine filename
        dir = os.path.dirname('__file__')
        f = os.path.join(dir, '..', 'data', filename)

        # load data
        with open(f, 'r') as fid:
            data = np.loadtxt(fid, delimiter=",")

        # separate features and labels
        self.X = data[:,:-1]
        self.y = data[:,-1]

    def plot(self, **kwargs):
        """Plot data."""

        if 'color' not in kwargs:
            kwargs['color'] = 'b'
        if 'marker' not in kwargs:
            kwargs['marker'] = 'o'
        if 'linestyle' not in kwargs:
            kwargs['linestyle'] = ''

        plt.plot(self.X, self.y, **kwargs)
        plt.xlabel('x', fontsize = 16)
        plt.ylabel('y', fontsize = 16)
        plt.show()

# wrapper functions around Data class
def load_data(filename):
    data = Data()
    data.load(filename)
    return data

def plot_data(X, y, **kwargs):
    data = Data(X, y)
    data.plot(**kwargs)

class PolynomialRegression:
    def __init__(self, m=1, reg_param=0):
        """
        Ordinary least squares regression.

        Attributes
        --------------------
            coef_   -- numpy array of shape (d,)
                       estimated coefficients for the linear regression problem
            m_      -- integer
                       order for polynomial regression
            lambda_ -- float
                       regularization parameter
        """
        self.coef_ = None
        self.m_ = m
        self.lambda_ = reg_param

    def generate_polynomial_features(self, X):
        """
        Maps X to an mth degree feature vector e.g. [1, X, X^2, ..., X^m].

        Parameters
        --------------------
            X       -- numpy array of shape (n,1), features

        Returns
        --------------------
            Phi     -- numpy array of shape (n,(m+1)), mapped features
        """

        n,d = X.shape

        ### ========== TODO: START ========== ###
        # part b: modify to create matrix for simple linear model
        # part g: modify to create matrix for polynomial model
        Phi = np.concatenate([X**i for i in range(self.m_+1)], axis=1)
        ### ========== TODO: END ========== ###

        return Phi

    def fit_GD(self, X, y, eta=None,
                eps=0, tmax=10000, verbose=False):
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using least squares batch gradient descent.

        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
            eta     -- float, step size
            eps     -- float, convergence criterion
            tmax    -- integer, maximum number of iterations
            verbose -- boolean, for debugging purposes

        Returns
        --------------------
            self    -- an instance of self
        """
        if self.lambda_ != 0:
            raise Exception("GD with regularization not implemented")

        if verbose:
            plt.subplot(1, 2, 2)
            plt.xlabel('iteration')
            plt.ylabel(r'$J(\theta)$')
            plt.ion()
            plt.show()

        X = self.generate_polynomial_features(X) # map features
        n,d = X.shape
        self.coef_ = np.zeros(d)                 # coefficients
        err_list  = np.zeros((tmax,1))           # errors per iteration
        eta_input = eta

        # GD loop
        for t in range(tmax):
            ### ========== TODO: START ========== ###
            # part f: update step size
            # change the default eta in the function signature to 'eta=None'
            # and update the line below to your learning rate function
            if eta_input is None:
                eta = 1 / (1 + t)
            ### ========== TODO: END ========== ###

            ### ========== TODO: START ========== ###
            # part d: update theta (self.coef_) using one step of GD
            # hint: you can write simultaneously update all theta
            # using vector math
            y_pred = np.matmul(X, self.coef_)
            gradient = 2 * eta * np.matmul(X.T, (y_pred - y))
            self.coef_ = self.coef_ - gradient

            # track error
            # hint: you cannot use self.predict(...) to make the predictions
            err_list[t] = np.sum(np.power(y - y_pred, 2)) / n
            ### ========== TODO: END ========== ###

            # stop?
            if t > 0 and abs(err_list[t] - err_list[t-1]) <= eps:
                break

            # debugging
            if verbose:
                x = np.reshape(X[:,1], (n,1))
                cost = self.cost(x,y)
                plt.subplot(1, 2, 1)
                plt.cla()
                plot_data(x, y, marker='o', linestyle='', scalex=True)
                self.plot_regression(scalex=False, scaley=False)
                plt.subplot(1, 2, 2)
                plt.plot([t+1], [cost], 'bo')
                plt.suptitle('iteration: %d, cost: %f' % (t+1, cost))
                plt.draw()
                plt.pause(0.05) # pause for 0.05 sec

        return t + 1

    def fit(self, X, y, l2regularize=None ):
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using the closed form solution.

        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
            l2regularize    -- set to None for no regularization. set to positive double for L2 regularization

        Returns
        --------------------
            self    -- an instance of self
        """

        X = self.generate_polynomial_features(X) # map features

        ### ========== TODO: START ========== ###
        # part e: implement closed-form solution
        # hint: use np.dot(...) and np.linalg.pinv(...)
        #       be sure to update self.coef_ with your solution
        self.coef_ = np.linalg.pinv(X.T @ X) @ X.T @ y
        ### ========== TODO: END ========== ###

    def predict(self, X):
        """
        Predict output for X.

        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features

        Returns
        --------------------
            y       -- numpy array of shape (n,), predictions
        """
        if self.coef_ is None:
            raise Exception("Model not initialized. Perform a fit first.")

        X = self.generate_polynomial_features(X) # map features

        ### ========== TODO: START ========== ###
        # part c: predict y
        y = np.matmul(X, self.coef_)
        ### ========== TODO: END ========== ###

        return y

    def cost(self, X, y):
        """
        Calculates the objective function.

        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets

        Returns
        --------------------
            cost    -- float, objective J(theta)
        """
        ### ========== TODO: START ========== ###
        # part d: compute J(theta)
        # J(theta) = sum((w^T X - y)^2)
        X = self.generate_polynomial_features(X)
        cost = np.sum(np.power(np.matmul(X, self.coef_) - y, 2))
        ### ========== TODO: END ========== ###
        return cost

    def rms_error(self, X, y):
        """
        Calculates the root mean square error.

        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets

        Returns
        --------------------
            error   -- float, RMSE
        """
        ### ========== TODO: START ========== ###
        # part h: compute RMSE
        N, d = X.shape
        error = np.sqrt(self.cost(X, y) / N)
        ### ========== TODO: END ========== ###
        return error

    def plot_regression(self, xmin=0, xmax=1, n=50, **kwargs):
        """Plot regression line."""
        if 'color' not in kwargs:
            kwargs['color'] = 'r'
        if 'linestyle' not in kwargs:
            kwargs['linestyle'] = '-'
        if 'marker' not in kwargs:
            kwargs['marker'] = ''

        X = np.reshape(np.linspace(0,1,n), (n,1))
        y = self.predict(X)
        plot_data(X, y, **kwargs)
        plt.show()

######################################################################
# main
######################################################################

def show(model, *data_list):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    plt.subplot(1, 1, 1)
    for i, data in enumerate(data_list):
        # abuse interactive plotting mode to
        # combine the graphs into one subplot
        plt.ion()
        data.plot(color=colors[i])
        plt.ioff()
    model.plot_regression()
    plt.show()

class TestLinearRegression(unittest.TestCase):
    def setUp(self):
        self.data = load_data('regression_train.csv')
        self.model = PolynomialRegression(m=1)

    def test_cost(self):
        self.model.coef_ = np.zeros(2)
        cost = self.model.cost(self.data.X, self.data.y)
        self.assertAlmostEqual(cost, 40.234, places=3)

    @unittest.skip
    def test_fit_GD(self):
        etas = [10**-4, 10**-3, 10**-2, 0.0407]
        for eta in etas:
            n = self.model.fit_GD(self.data.X, self.data.y, eta=eta)
            w = self.model.coef_
            cost = self.model.cost(self.data.X, self.data.y)
            print(f'eta: {eta}, iterations: {n}, coef: {w}, cost: {cost}')

    @unittest.skip
    def test_fit(self):
        self.model.fit(self.data.X, self.data.y)
        show(self.model, self.data)

    @unittest.skip
    def test_timed_comparison(self):
        fit = partial(self.model.fit, self.data.X, self.data.y)
        fit_GD = partial(self.model.fit_GD, self.data.X, self.data.y, eta=0.01)
        print('timing closed form', flush=True)
        print(timeit.timeit(fit, number=10000) / 10000)
        print('timing GD', flush=True)
        print(timeit.timeit(fit_GD, number=10000) / 10000)

    def test_dynamic_learning_rate(self):
        n = self.model.fit_GD(self.data.X, self.data.y, eta=None, verbose=False)
        print(n)

class TestPolynomialRegression(unittest.TestCase):
    def setUp(self):
        self.data = load_data('regression_train.csv')
        self.model = PolynomialRegression(m=10)

    @unittest.skip
    def test_polynomial_regression_gd(self):
        self.model.fit_GD(self.data.X, self.data.y, eta=0.01, verbose=True)
        show(model, data)

    @unittest.skip
    def test_polynomial_regression(self):
        self.model.fit(self.data.X, self.data.y)
        show(model, data)

class TestFindBestM(unittest.TestCase):
    def setUp(self):
        self.train = load_data('regression_train.csv')
        self.test = load_data('regression_test.csv')

    @unittest.skip
    def test_find_best_m(self):
        train_errors = []
        test_errors = []
        for m in range(11):
            model = PolynomialRegression(m=m)
            model.fit(self.train.X, self.train.y)
            #show(model, self.train, self.test)
            train_error = model.rms_error(self.train.X, self.train.y)
            test_error = model.rms_error(self.test.X, self.test.y)
            train_errors.append(train_error)
            test_errors.append(test_error)

        xs = np.arange(0, 11)
        i = np.argmin(test_errors)
        print(i)
        print(test_errors[i])
        print(train_errors[i])

        fig, ax = plt.subplots()
        ax.plot(xs, train_errors, label='Train RMSE')
        ax.plot(xs, test_errors, color='g', label='Test RMSE')
        fig.suptitle('Finding Best High Order Polynomial Fit')
        ax.set_ylabel('RSME')
        ax.set_xlabel(r'Polynomial Degree ($m$)')
        ax.legend()  # sending it since 1996
        ax.grid()
        plt.show()

def plot_train_test():
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    ax1.set_title('Train Data')
    ax2.set_title('Test Data')
    train = load_data('regression_train.csv')
    test = load_data('regression_test.csv')
    ax1.scatter(train.X, train.y)
    ax2.scatter(test.X, test.y, color='g')
    plt.show()

if __name__ == "__main__":
    #plot_train_test()
    unittest.main()
