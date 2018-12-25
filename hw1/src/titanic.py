"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Titanic
"""

# Use only the provided packages!
import math
import csv
from util import *
from collections import Counter, defaultdict

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn import metrics

######################################################################
# classes
######################################################################

class Classifier(object) :
    """
    Classifier interface.
    """

    def fit(self, X, y):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()

class MajorityVoteClassifier(Classifier) :

    def __init__(self) :
        """
        A classifier that always predicts the majority class.

        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None

    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """
        majority_val = Counter(y).most_common(1)[0][0]
        self.prediction_ = majority_val
        return self

    def predict(self, X) :
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")

        n,d = X.shape
        y = [self.prediction_] * n
        return y

class RandomClassifier(Classifier) :

    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.

        Attributes
        --------------------
            probabilities_ -- class distribution dict (key = class, val = probability of class)
        """
        self.probabilities_ = None

    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """

        ### ========== TODO: START ========== ###
        # part b: set self.probabilities_ according to the training set
        freqs = Counter(y)
        total = sum(freqs.values())
        self.probabilities_ = {elem: freq/total for elem, freq in freqs.items()}
        ### ========== TODO: END ========== ###

        return self

    def predict(self, X, seed=1234) :
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)

        ### ========== TODO : START ========== ###
        # part b: predict the class for each test example
        # hint: use np.random.choice (be careful of the parameters)
        n, d = X.shape
        y = np.random.choice(
            list(self.probabilities_.keys()),
            size=(n,),
            p=list(self.probabilities_.values())
        )
        ### ========== TODO : END ========== ###

        return y

######################################################################
# functions
######################################################################
def plot_histograms(X, y, Xnames, yname) :
    n,d = X.shape  # n = number of examples, d =  number of features
    fig = plt.figure(figsize=(20,15))
    nrow = 3; ncol = 3
    for i in range(d) :
        fig.add_subplot (3,3,i)
        data, bins, align, labels = plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname, show = False)
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xnames[i])
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')

    plt.savefig ('histograms.pdf')

def plot_histogram(X, y, Xname, yname, show = True) :
    """
    Plots histogram of values in X grouped by y.

    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """

    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in range(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))

    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = list(range(int(math.floor(min(features))), int(math.ceil(max(features)))+1))
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'

    # plot
    if show == True:
        plt.figure()
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
        plt.show()

    return data, bins, align, labels

def error(clf, X, y, ntrials=100, test_size=0.2) :
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.

    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials

    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """

    ### ========== TODO : START ========== ###
    # compute cross-validation error over ntrials
    # hint: use train_test_split (be careful of the parameters)
    train_error = 0
    test_error = 0

    for trial_i in range(ntrials):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=trial_i,
        )
        fitted = clf.fit(X_train, y_train)
        train_error += getError(y_train, fitted.predict(X_train))
        test_error += getError(y_test, fitted.predict(X_test))

    train_error /= ntrials
    test_error /= ntrials
    ### ========== TODO : END ========== ###

    return train_error, test_error


def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(list(zip(y_pred)))
    out.close()

######################################################################
# main
######################################################################
def getError(y, y_pred):
    return 1 - metrics.accuracy_score(y, y_pred, normalize=True)

def testClassifier(clf, X, y):
    clf.fit(X, y)
    y_pred = clf.predict(X)
    train_error = getError(y, y_pred)
    print('\t-- training error: {:.3f}'.format(train_error))

def findBestK(X, y):
    scores = []
    folds = 10

    for k in range(1,50,2):
        clf = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree')
        fold_results = cross_val_score(clf, X, y, cv=folds)
        scores.append(1 - sum(fold_results)/folds)

    plt.figure()
    plt.plot(list(range(1,50,2)), scores)
    plt.grid()
    plt.title('Finding Best K of Nearest Neighbors Classifier')
    plt.ylabel('mean error')
    plt.xlabel('n_neighbors (k)')
    plt.xticks(list(range(1,51,2)))
    plt.yticks(np.arange(0.28, 0.35, 0.005))
    plt.show()

    best_k = 2 * (np.argmin(scores) + 1) - 1
    print(f'best k: {best_k}')
    return best_k

def findBestDepth(X, y):
    test_scores = []
    train_scores = []
    #folds = 10

    for depth in range(1, 21):
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=depth)
        train_error, test_error = error(clf, X, y)
        print(f'{depth} {train_error} {test_error}')
        train_scores.append(train_error)
        test_scores.append(test_error)

        #fold_results = cross_validate(
        #    clf, X, y, cv=folds,
        #    return_train_score=True
        #)
        #test_scores.append(1 - sum(fold_results['test_score'])/folds)
        #train_scores.append(1 - sum(fold_results['train_score'])/folds)

    plt.figure()
    plt.plot(list(range(1,21)), train_scores, label='train scores')
    plt.plot(list(range(1,21)), test_scores, label='test scores')
    plt.grid()
    plt.legend()
    plt.title('Finding Best Depth of Decision Tree Classifier')
    plt.ylabel('mean error')
    plt.xlabel('depth')
    plt.xticks(list(range(1,21)))
    plt.show()

    best_depth = np.argmin(test_scores) + 1
    print(f'best depth: {best_depth}')
    return best_depth

def plotLearningCurve(X, y, n_neighbors, depth):
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    clfs = [
        DecisionTreeClassifier(criterion='entropy', max_depth=depth),
        KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='kd_tree')
    ]
    n_examples = X.shape[0]
    plt.figure()

    #for clf in clfs:
    #    test_errors = []
    #    train_errors = []
    #    train_fractions = np.linspace(0.1, 1, 10)
    #    for train_size in train_fractions:
    #        #if train_size != 1.0:
    #        #    X_train, _, y_train, _ = train_test_split(X, y, train_size=train_size, random_state=0)
    #        #else:
    #        #    X_train, y_train = X, y
    #        X_train = X[:int(n_examples*train_size)]
    #        y_train = y[:int(n_examples*train_size)]

    #        clf.fit(X_train, y_train)
    #        test_errors.append(getError(y_test, clf.predict(X_test)))
    #        train_errors.append(getError(y_train, clf.predict(X_train)))

    #    plt.plot(train_fractions, train_errors, label=f'{type(clf).__name__} training error')
    #    plt.plot(train_fractions, test_errors, label=f'{type(clf).__name__} test error')

    test_errors = defaultdict(list)
    train_errors = defaultdict(list)
    train_fractions = np.linspace(0.1, 1, 10)

    for train_size in train_fractions:
        X_train = X[:int(n_examples*train_size)]
        y_train = y[:int(n_examples*train_size)]

        for clf in clfs:
            clf.fit(X_train, y_train)
            test_errors[clf].append(getError(y_test, clf.predict(X_test)))
            train_errors[clf].append(getError(y_train, clf.predict(X_train)))

    for clf in clfs:
        plt.plot(train_fractions, train_errors[clf], label=f'{type(clf).__name__} training error')
        plt.plot(train_fractions, test_errors[clf], label=f'{type(clf).__name__} test error')

    plt.grid()
    plt.ylabel('mean error')
    plt.xlabel('fraction of training data')
    plt.xticks(train_fractions)
    plt.title('Learning Curves')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.show()


def main():
    # load Titanic dataset
    titanic = load_data("titanic_train.csv", header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n, d = X.shape # n examples, d features per example


    #========================================
    # part a: plot histograms of each feature
    #print('Plotting...')
    #for i in range(d) :
    #    plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)

    #========================================
    # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    testClassifier(MajorityVoteClassifier(), X, y)

    ### ========== TODO : START ========== ###
    # part b: evaluate training error of Random classifier
    print('Classifying using Random...')
    testClassifier(RandomClassifier(), X, y)
    ### ========== TODO : END ========== ###

    ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain
    print('Classifying using Decision Tree...')
    testClassifier(DecisionTreeClassifier(criterion='entropy'), X, y)
    ### ========== TODO : END ========== ###

    # note: uncomment out the following lines to output the Decision Tree graph
    """
    # save the classifier -- requires GraphViz and pydot
    import StringIO, pydot
    from sklearn import tree
    dot_data = StringIO.StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=Xnames)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("dtree.pdf")
    """

    ### ========== TODO : START ========== ###
    # part d: evaluate training error of k-Nearest Neighbors classifier
    # use k = 3, 5, 7 for n_neighbors
    print('Classifying using k-Nearest Neighbors (k=3)...')
    testClassifier(KNeighborsClassifier(n_neighbors=3, algorithm='kd_tree'), X, y)
    print('Classifying using k-Nearest Neighbors (k=5)...')
    testClassifier(KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree'), X, y)
    print('Classifying using k-Nearest Neighbors (k=7)...')
    testClassifier(KNeighborsClassifier(n_neighbors=7, algorithm='kd_tree'), X, y)
    ### ========== TODO : END ========== ###

    ### ========== TODO : START ========== ###
    # part e: use cross-validation to compute average training and test error of classifiers
    print('Investigating various classifiers...')
    classifiers = [
        MajorityVoteClassifier(),
        RandomClassifier(),
        DecisionTreeClassifier(criterion='entropy'),
        KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree'),
    ]
    for clf in classifiers:
        train_error, test_error = error(clf, X, y)
        print(f'{type(clf).__name__}:')
        print(f'\ttrain error: {train_error:.3f}')
        print(f'\ttest error: {test_error:.3f}')
    ### ========== TODO : END ========== ###

    ### ========== TODO : START ========== ###
    # part f: use 10-fold cross-validation to find the best value of k for k-Nearest Neighbors classifier
    print('Finding the best k for KNeighbors classifier...')
    best_k = findBestK(X, y)
    ### ========== TODO : END ========== ###

    ### ========== TODO : START ========== ###
    # part g: investigate decision tree classifier with various depths
    print('Investigating depths...')
    #best_depth = findBestDepth(X, y)
    ### ========== TODO : END ========== ###

    ### ========== TODO : START ========== ###
    # part h: investigate Decision Tree and k-Nearest Neighbors classifier with various training set sizes
    print('Investigating training set sizes...')
    plotLearningCurve(X, y, n_neighbors=best_k, depth=6) # TODO
    ### ========== TODO : END ========== ###

    print('Done')

if __name__ == "__main__":
    main()
