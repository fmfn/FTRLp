from __future__ import division
from __future__ import print_function

"""
 ------------ Follow The Regularized Leader - Proximal ------------

FTRL-P is an online classification algorithm that combines both L1 and L2
norms, particularly suited for large data sets with extremely high dimensionality.

This implementation follow the algorithm by H. B. McMahan et. al. It minimizes
the LogLoss function iteratively with a combination of L2 and L1 (centralized
at the current point) norms and adaptive, per coordinate learning rates.

This algorithm is efficient at obtaining sparsity and has proven to perform
very well in massive Click-Through-Rate prediction tasks.

This module contains two objects...

References:
    * Follow-the-Regularized-Leader and Mirror Descent: Equivalent Theorems
      and L1 Regularization, H. Brendan McMahan
    * Ad Click Prediction: a View from the Trenches, H. Brendan McMahan et. al.

"""

from math import log, exp, fabs, sqrt
from csv import DictReader
from datetime import datetime
from random import random


def log_loss(y, p):
    """
    --- Log_loss computing function
    A function to compute the log loss of a predicted probability p given
    a true target y.

    :param y: True target value
    :param p: Predicted probability
    :return: Log loss.
    """
    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1 else -log(1. - p)


class DataGen(object):
    """
    DataGen is an object to generate the data that is fed to the
    classifier.

    It reads the data file one row at a time, hashes it
    and returns it.

    The names and types of columns must be passed to it, so that categorical,
    target, numerical and identification columns can be treated differently.

    It also keeps track of the name and position of all features to allow
    the classifier to keep track of the coefficients by feature.
    """

    def __init__(self, max_features, target, descriptive=(), categorical=(), numerical=None, transformation=None):
        """
        The object initialized with the maximum number of features to be generated and the
        names of the appropriate columns.

        Categorical columns are hashed while numerical columns are kept as is, therefore
        care must be taken with normalization and pre processing.

        :param max_features: The maximum number of features to generate. It includes all
                             numerical and categorical features. Must be greater than the
                             number of numerical features.

        :param target: The name of the target variable. It must be a binary variable taking
                       values in {0, 1}.

        :param descriptive: Descriptive features that are used to identify the samples but
                            are not to be used for modelling, such as IDs, public
                            identifiers, etc.

        :param categorical: Categorical variable to be hashed.

        :param numerical: Numerical variable. These will not be hashed but will be used in
                          the modelling phase.

        """

        # --- Instance variables.
        # Instance variables are created for columns names and the number of numerical
        # columns in addition to all of the object's parameters.

        # Stores the maximum number of features to generate while hashing
        self.mf = max_features

        # Stores the name of the target variable.
        self.y = target

        # Stores a list with the names of all descriptive variables.
        self.ids = descriptive

        # Stores a list with the names of all categorical variables.
        self.cat = categorical

        # Stores a list with the names of all numerical variables.
        self.num = numerical

        # Stores a dictionary with the names of numerical variable to apply a given function to.
        self.tra = transformation if transformation is not None else {}

        # Dictionary to store names
        self.names = {}

        # --- Numerical features
        # Numerical features are indexed in sorted order. The number
        # of features is determined by the variable size. The value
        # of each feature is just the value read from the file. Start
        # by defining what is numeric. If the user does not pass the
        # names of all numerical features, the code will assume
        # every columns that is not id, target or categorical is
        # numeric and find their name when the training process begin.
        if self.num is not None:
            self.num_cols = sorted(self.num)

            # Store the names in our names dictionary
            self.names.update(dict(zip(self.num_cols, range(len(self.num_cols)))))
        else:
            self.num_cols = []


        # --- Something to build model on
        # Make sure the user passed some information on the columns to
        # be used to build the model upon
        assert len(self.cat) + len(self.num_cols) > 0, 'At least one categorical or numerical feature must ' \
                                                       'be provided.'

    def _fetch(self, path):
        """
        This method is the core reason this object exists. It is a python generator
        that hashes categorical variables, combines them to numerical variables and
        yields all the relevant information, row by row.

        :param path: Path of the data file to be read.

        :return: YIELDS the current row, ID information, feature values and the target value.
                 even if the file does not contain a target field it returns a target value
                 of zero anyway.
        """

        for t, row in enumerate(DictReader(open(path))):
            # --- Variables
            #   t: The current line being read
            # row: All the values in this line

            # --- Ids and other descriptive fields
            # Process any descriptive fields and put it all in a list.
            ids = []
            for ID in self.ids:
                ids.append(row[ID])
                del row[ID]

            # --- Target
            # Process target and delete its entry from row if it exists
            # otherwise just ignore and move along
            y = 0.
            if self.y in row:
                if row[self.y] == '1':
                    y = 1.
                del row[self.y]

            # --- Features
            # Initialize an empty dictionary to hold feature
            # indexes and their corresponding values.
            #
            x = {}

            # --- Enough features?
            # For the very first row make sure we have enough features (max features
            # is large enough) by computing the number of numerical columns and
            # asserting that the maximum number of features is larger than it.
            if t == 0:
                # --- Hash size
                # Computes a constant to add to hash index, it dictates the
                # number of features that will not be hashed
                num_size = len(self.num_cols)
                size = num_size + len(self.tra)

                # Make sure there is enough space for hashing
                assert self.mf > size, 'Not enough dimensions to fit all features.'

            # --- Numerical Variables
            # Now we loop over numerical variables
            for i, key in enumerate(self.num_cols):
                # --- No transformation
                # If no transformation is necessary, just store the actual value
                # of the variable.
                x[i] = float(row[key])

            # --- Transformations
            # Create on the fly transformed variables. The user passes a map of the
            # name of the new variable to a tuple containing the name of the original
            # variable to be transformed and the function to be applied to it.
            # Once completed the new name is appended to the names dictionary with its
            # corresponding index.#
            for i, key in enumerate(self.tra):
                # Start by addition to the data array x the new transformed values
                # by looping over new_features and applying the transformation to the
                # desired old feature.
                x[num_size + i] = self.tra[key][1](row[self.tra[key][0]])

                # Create a key in names dictionary with the new name and its
                # corresponding index.
                self.names[key] = num_size + i

            # --- Categorical features
            # Categorical features are hashed. For each different kind a
            # hashed index is created and a value of 1 is 'stored' in that
            # position.
            for key in self.cat:
                # --- Category
                # Get the categorial variable from row
                value = row[key]

                # --- Hash
                # One-hot encode everything with hash trick
                index = (abs(hash(key + '_' + value)) % (self.mf - size)) + size
                x[index] = 1.

                # --- Save Name
                # Save the name and index to the names dictionary if its a new feature
                # AND if there's still enough space.
                if key + '_' + value not in self.names and len(self.names) < self.mf:
                    self.names[key + '_' + value] = index

            # Yield everything.
            yield t, ids, x, y

    def train(self, path):
        """
        The train method is just a wrapper around the _fetch generator to comply
        with sklearn's API.

        :param path: The path for the training file.

        :return: YIELDS row, features, target value
        """

        # --- Generates train data
        # This is just a generator on top of the basic _fetch. If this was python 3 I
        # could use 'yield from', but I don't think this syntax exists in python 2.7,
        # so I opted to use the explicit, less pythonic way.
        for t, ids, x, y in self._fetch(path):
            # --- Variables
            #   t: Current row
            # ids: List of ID information
            #   x: Feature values
            #   y: Target values

            yield t, x, y

    def test(self, path):
        """
        The test method is just a wrapper around the _fetch generator to comply
        with sklearn's API.

        :param path: The path for the test file.

        :return: YIELDS row, features
        """

        # --- Generates test data
        # This is just a generator on top of the basic _fetch. If this was python 3 I
        # could use 'yield from', but I don't think this syntax exists in python 2.7,
        # so I opted to use the explicit, less pythonic way.
        for t, ids, x, y in self._fetch(path):
            # --- Variables
            #   t: Current row
            # ids: List of ID information
            #   x: Feature values
            #   y: Target values

            yield t, x


class FTRLP(object):
    """
    --- Follow The Regularized Leader - Proximal ---

    FTRL-P is an online classification algorithm that combines both L1 and L2
    norms, particularly suited for large data sets with extremely high dimensionality.

    This implementation follow the algorithm by H. B. McMahan et. al. It minimizes
    the LogLoss function iteratively with a combination of L2 and L1 (centralized
    at the current point) norms and adaptive, per coordinate learning rates.

    This algorithm is efficient at obtaining sparsity and has proven to perform
    very well in massive Click-Through-Rate prediction tasks.

    References:
        * Follow-the-Regularized-Leader and Mirror Descent: Equivalent Theorems
          and L1 Regularization, H. Brendan McMahan
        * Ad Click Prediction: a View from the Trenches, H. Brendan McMahan et. al.
    """

    def __init__(self, alpha=1, beta=1, l1=1, l2=1, subsample=1, epochs=1, rate=0):
        """
        Initializes the classifier's learning rate constants alpha and beta,
        the regularization constants L1 and L2, and the maximum number of
        features (limiting factor of the hash function).

        The per feature learning rate is given by:
                eta = alpha / ( beta + sqrt( sum g**g ) )

        :param alpha: Learning rate's proportionality constant.

        :param beta: Learning rate's parameter.

        :param l1: l1 regularization constant.

        :param l2: l2 regularization constant.

        :return:
        """

        # --- Classifier Parameters
        # The FTRLP algorithm has four free parameters that can be tuned as pleased.

        # Learning rate's proportionality constant.
        self.alpha = alpha
        # Learning rate's parameter.
        self.beta = beta
        # L1 regularization constant.
        self.l1 = l1
        # L2 regularization constant.
        self.l2 = l2

        # --- Log likelihood
        # Stores the log likelihood during the whole
        # fitting process.
        self.log_likelihood_ = 0
        self.loss = []

        # --- Weight parameters.
        # Lists and dictionaries to hold the weights. Initiate
        # the weight vector z and learning rate n as None so that
        # when self.train is called multiple times it will not
        # overwrite the stored values. This essentially allows epoch
        # training to take place, albeit a little bit ugly.
        self.z = None
        self.n = None
        # The weight vector used for prediction is constructed on the fly
        # and, in order to keep the memory cost low, it is a dictionary
        # that receives values and keys as needed.

        # --- Coefficients
        # Lists to store the coefficients and their corresponding names.
        # Initialized to None and constructed once the training method is
        # completed. In case of multiple epochs, these quantities will be
        # computed multiple times.
        self.coef_ = {}
        self.cname = None

        # --- Target Ratio
        # Store the ratio of each class of a binnary target variable to use
        # it to make weighted discrete label predictions.
        self.target_ratio = 0.

        # --- Printing Rate
        # Number of samples to train and predict on before printing
        # current status
        self.rate = rate

        # --- Subsample
        # While online methods can't be shuffle, combining subsampling of
        # the training set with multiple epoch training gives similar results.
        self.subsample = subsample

        # --- Epochs
        # something...
        self.epochs = epochs

        # --- Flag for partial fit
        # Keeps a flag to allow the user to train multiple times
        # without overwriting the object.
        self.fit_flag = False

    def _build_p(self, data_gen, path):
        # Maybe is worth migrating the weight construction algorithm
        # to here, I think it could clean up the code a little a bit
        # in both train and predict methods.
        pass

    def _clear_params(self):
        """
        If the fit method is called multiple times, all trained parameters
        must be cleared allowing for a fresh start. This function simply
        resets everything back to square one.

        :return: Nothing
        """

        # All models parameters are set to their original value (see
        # __init__ description
        self.log_likelihood_ = 0
        self.loss = []
        self.z = None
        self.n = None
        self.coef_ = {}
        self.cname = None

    def get_params(self, deep=True):
        """
        A function to return a map of parameters names and values.

        :param deep: Not sure yet, gotta check sklearn usage.

        :return: Dictionary mapping parameters names to their values
        """

        ps = {'alpha': self.alpha,
              'beta': self.beta,
              'l1': self.l1,
              'l2': self.l2,
              'subsample': self.subsample,
              'epochs': self.epochs,
              'rate': self.rate}

        return ps

    def set_params(self, **params):
        """


        :param params:
        :return:
        """

        for key, value in params.iteritems():
            setattr(self, key, value)

    def _update(self, y, p, x, w):
        """
        # --- Update weight vector and learning rate.
        # With the prediction round completed we can proceed to
        # updating the weight vector z and the learning rate eta
        # based on the last observed label.

        # To do so we will use the computed probability and target
        # value to find the gradient loss and continue from there.

        # The gradient for the log likelihood for round t can easily
        # be shown to be:
        #               g_i = (p - y) * x_i, (round t)

        # The remaining quantities are updated according to the
        # minimization procedure outlined in [2].

        :param y: True target variable
        :param p: Predicted probability for the current sample
        :param x: Non zero feature values
        :param w: Weights
        :return: Nothing
        """

        # --- Update loop
        # Loop over all relevant indexes and update all values
        # accordingly.
        for i in x.keys():
            # --- Compute Gradient of LogLoss
            g = (p - y) * x[i]

            # --- Update constant sigma
            # Note that this upgrade is equivalent to
            #       (eta_(t, i))^-1 - (eta_(t - 1, i))^-1
            # as discussed in [2].
            s = (sqrt(self.n[i] + g * g) - sqrt(self.n[i])) / self.alpha

            # --- Increment changes
            # Finally,  increment the appropriate changes to weights and
            # learning rate vectors.
            self.z[i] += g - s * w[i]
            self.n[i] += g * g

    def _train(self, data_gen, path):
        """
        --- Fitting method ---

        Online fitting method. It takes one sample at a time, builds
        the weight vector on the fly and computes the dot product of
        weight vector and values and a prediction is made.

        Then the true label of the target variable is observed and the
        loss is added.

        Once this is completed the weights are updated based on the
        previously observed values.

        :param data_gen: An instance of the DataGen class
        :param path: The path to the training set
        :return:
        """

        # Best way? Proper coding means no access to protected members...
        if self.z is None and self.n is None:
            self.z = [0.] * data_gen.mf
            self.n = [0.] * data_gen.mf

        # --- Start the clock!
        start_time = datetime.now()

        for t, x, y in data_gen.train(path):
            # --- Variables
            #   t: Current row
            #   x: Feature values
            #   y: Target values

            # --- Target Ratio Update
            # Rolling calculation of the target average
            self.target_ratio = (1.0 * (t * self.target_ratio + y)) / (t + 1)

            # --- Stochastic sample selection
            # Chose whether or not to use a sample in
            # training time. Since online methods can't
            # really be shuffle we can use this combined
            # with multiple epochs to create heterogeneity.
            #if random() > self.subsample and ((t + 1) % self.rate != 0):
            if random() > self.subsample and (t + 1) % self.rate != 0:
                continue

            # --- Dot product init.
            # The dot product is computed as the weights are calculated,
            # here it is initiated at zero.
            wtx = 0

            # --- Real time weights
            # Initialize an empty dictionary to hold the weights
            w = {}

            # --- Weights and prediction
            # Computes the weights for numerical features using the
            # indexes and values present in the x dictionary. And make
            # a prediction.

            # This first loop build the weight vector on the fly. Since
            # we expect most weights to be zero, the weight vector can
            # be constructed in real time. Furthermore, there is no
            # reason to store it, neither to clear it, since at each
            # iteration only the relevant indexes are populated and used.
            for indx in x.keys():
                # --- Loop over indicator I
                # x.keys() carries all the indexes of the feature
                # vector with non-zero entries. Therefore, we can
                # simply loop over it since anything else will not
                # contribute to the dot product w.x, and, consequently
                # to the prediction.
                if fabs(self.z[indx]) <= self.l1:
                    # --- L1 regularization
                    # If the condition on the absolute value of the
                    # vector Z is not met, the weight coefficient is
                    # set exactly to zero.
                    w[indx] = 0
                else:
                    # --- Non zero weight
                    # Provided abs(z_i) is large enough, the weight w_i
                    # is computed. First, the sign of z_i is determined.
                    sign = 1. if self.z[indx] >= 0 else -1.

                    # Then the value of w_i if computed and stored. Note
                    # that any previous value w_i may have had will be
                    # overwritten here. Which is fine since it will not
                    # be used anywhere outside this (t) loop.
                    w[indx] = - (self.z[indx] - sign * self.l1) / \
                                (self.l2 + (self.beta + sqrt(self.n[indx])) / self.alpha)

                # --- Update dot product
                # Once the value of w_i is computed we can use to compute
                # the i-th contribution to the dot product w.x. Which, here
                # is being done inside the index loop, compute only coordinates
                # that could possible be non-zero.
                wtx += w[indx] * x[indx]

            # --- Make a prediction
            # With the w.x dot product in hand we can compute the output
            # probability by putting wtx through the sigmoid function.
            # We limit wtx value to lie in the [-35, 35] interval to
            # avoid round off errors.
            p = 1. / (1. + exp(-max(min(wtx, 35.), -35.)))

            # --- Update the loss function
            # Now we look at the target value and use it, together with the
            # output probability that was just computed to find the loss we
            # suffer this round.
            self.log_likelihood_ += log_loss(y, p)

            # --- Verbose section
            if (self.rate > 0) and (t + 1) % self.rate == 0:
                # Append to the loss list.
                self.loss.append(self.log_likelihood_)

                # Print all the current information
                print('Training Samples: {0:9} | '
                      'Loss: {1:11.2f} | '
                      'Time taken: {2:4} seconds'.format(t + 1,
                                                         self.log_likelihood_,
                                                         (datetime.now() - start_time).seconds))

            # --- Update weights
            # Finally, we now how well we did this round and move on to
            # updating the weights based on the current status of our
            # knowledge.
            self._update(y, p, x, w)

        # --- Coefficient names and indexes
        # Bind the feature names to their corresponding coefficient obtained from
        # the regression.
        self.coef_.update(dict([[key, self.z[data_gen.names[key]]] for key in data_gen.names.keys()]))


    def fit(self, data_gen, path):
        """
        Epoch wrapper around the main fitting method _train

        :param data_gen: An instance of the DataGen class
        :param path: The path to the training set
        :return:
        """

        # --- Check fit flag
        # Make sure the fit methods is starting from a clean slate by
        # checking the fit_flag variable and calling the _clear_params
        # function if necessary.
        # While always calling _clear_params would do the job, by setting
        # this flag we are also able to call fit multiple times WITHOUT
        # clearing all parameters --- See partial_fit.
        if self.fit_flag:
            self._clear_params()

        # --- Start the clock!
        total_time = datetime.now()

        # Train epochs
        for epoch in range(self.epochs):
            # --- Start the clock!
            epoch_time = datetime.now()

            # --- Verbose
            # Print epoch if verbose is turned on
            if self.rate > 0:
                print('TRAINING EPOCH: {0:2}'.format(epoch + 1))
                print('-' * 18)

            self._train(data_gen, path)

            # --- Verbose
            # Print time taken if verbose is turned on
            if self.rate > 0:
                print('EPOCH {0:2} FINISHED IN {1} seconds'.format(epoch + 1,
                                                                   (datetime.now() - epoch_time).seconds))
                print()

        # --- Verbose
        # Print fit information if verbose is on
        if self.rate > 0:
            print(' --- TRAINING FINISHED IN '
                  '{0} SECONDS WITH LOSS {1:.2f} ---'.format((datetime.now() - total_time).seconds,
                                                             self.log_likelihood_))
            print()

        # --- Fit Flag
        # Set fit_flag to true. If fit is called again this is will trigger
        # the call of _clean_params. See partial_fit for different usage.
        self.fit_flag = True

    def partial_fit(self, data_gen, path):
        """
        Simple solution to allow multiple fit calls without overwriting
        previously calculated weights, losses and etc.

        :param data_gen: An instance of the DataGen class
        :param path: The path to the training set
        :return:
        """

        # --- Fit Flag
        # Start by reseting fit_flag to false to "trick"
        # the fit method into keep training without overwriting
        # previously calculated quantities.
        self.fit_flag = False

        # --- Fit
        # Call the fit method and proceed as normal
        self.fit(data_gen, path)

    def predict_proba(self, data_gen, path):
        """
        --- Predicting Probabilities method ---

        Predictions...

        :param data_gen: An instance of the DataGen class
        :param path: The path to the test set

        :return: A list with predicted probabilities
        """

        # --- Results
        # Initialize an empty list to hold predicted values.
        result = []

        # --- Start the clock!
        start_time = datetime.now()

        for t, x in data_gen.test(path):
            # --- Variables
            #   t: Current row
            #   x: Feature values

            # --- Dot product init.
            # The dot product is computed as the weights are calculated,
            # here it is initiated at zero.
            wtx = 0

            # --- Real time weights
            # Initialize an empty dictionary to hold the weights
            w = {}

            # --- Weights and prediction
            # Computes the weights for numerical features using the
            # indexes and values present in the x dictionary. And make
            # a prediction.

            # This first loop build the weight vector on the fly. Since
            # we expect most weights to be zero, the weight vector can
            # be constructed in real time. Furthermore, there is no
            # reason to store it, neither to clear it, since at each
            # iteration only the relevant indexes are populated and used.
            for indx in x.keys():
                # --- Loop over indicator I
                # x.keys() carries all the indexes of the feature
                # vector with non-zero entries. Therefore, we can
                # simply loop over it since anything else will not
                # contribute to the dot product w.x, and, consequently
                # to the prediction.
                if fabs(self.z[indx]) <= self.l1:
                    # --- L1 regularization
                    # If the condition on the absolute value of the
                    # vector Z is not met, the weight coefficient is
                    # set exactly to zero.
                    w[indx] = 0
                else:
                    # --- Non zero weight
                    # Provided abs(z_i) is large enough, the weight w_i
                    # is computed. First, the sign of z_i is determined.
                    sign = 1. if self.z[indx] >= 0 else -1.

                    # Then the value of w_i if computed and stored. Note
                    # that any previous value w_i may have had will be
                    # overwritten here. Which is fine since it will not
                    # be used anywhere outside this (t) loop.
                    w[indx] = - (self.z[indx] - sign * self.l1) / \
                                (self.l2 + (self.beta + sqrt(self.n[indx])) / self.alpha)

                # --- Update dot product
                # Once the value of w_i is computed we can use to compute
                # the i-th contribution to the dot product w.x. Which, here
                # is being done inside the index loop, compute only coordinates
                # that could possible be non-zero.
                wtx += w[indx] * x[indx]

            # --- Make a prediction
            # With the w.x dot product in hand we can compute the output
            # probability by putting wTx through the sigmoid function.
            # We limit wTx value to lie in the [-35, 35] interval to
            # avoid round off errors.
            result.append(1. / (1. + exp(-max(min(wtx, 35.), -35.))))

            # Verbose section - Still needs work...
            if (t + 1) % self.rate == 0:
                # print some stuff
                print('Test Samples: {0:8} | '
                      'Time taken: {1:3} seconds'.format(t + 1,
                                                         (datetime.now() - start_time).seconds))

        # All done, return the predictions!
        return result

    def predict(self, data_gen, path):
        """
        --- Predicting method ---

        Predictions...

        :param data_gen: An instance of the DataGen class
        :param path: The path to the test set

        :return: A list with predicted probabilities
        """

        # --- Probabilities
        # Compute probabilities by invoking the predict_proba method
        probs = self.predict_proba(data_gen, path)

        # --- Return
        # Return binary labels. The threshold is set using the mean value of the
        # target variable.
        return map(lambda x: 0 if x <= self.target_ratio else 1, probs)