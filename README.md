[![Code Health](https://landscape.io/github/fmfn/FTRLp/master/landscape.svg?style=flat)](https://landscape.io/github/fmfn/FTRLp/master)

 ------------ Follow The Regularized Leader - Proximal ------------

FTRL-P is an online classification algorithm that combines both L1 and L2 regularization and feature based, adaptive 
learning rates, particularly suited for large data sets with extremely high dimensionality and multi-level categorical 
features.

This implementation follow the algorithm by H. B. McMahan et. al. It minimizes the LogLoss function iteratively with a 
combination of L2 and L1 (centralized at the current point) norms and adaptive, per coordinate learning rates. This is a
pure Python implementation --- no dependencies are needed ---, designed to be very memory efficient and suitable for 
 arbitrarily large data sets.

More over this method is very efficient at obtaining sparsity and has proven to perform very well in massive 
Click-Through-Rate prediction tasks.


** Data handling

To allow for memory efficiency regardless of the size of the dataset being handled, the classifier is fed one sample at
 a time by a python generator. This generator is implemented in the DataGen class together with a couple of other useful
 methods.

** FTRLP classifier

The main object...




References:
* Follow-the-Regularized-Leader and Mirror Descent: Equivalent Theorems and L1 Regularization, H. Brendan McMahan
* Ad Click Prediction: a View from the Trenches, H. Brendan McMahan et. al.
