from sklearn import linear_model
from scipy import stats
import numpy as np
from scipy.stats import t


class LogisticRegression(linear_model.LogisticRegression):
    """
    LinearRegression class after sklearn's, but calculate t-statistics
    and p-values for model coefficients (betas).
    Additional attributes available after .fit()
    are `t` and `p` which are of the shape (y.shape[1], X.shape[1])
    which is (n_features, n_coefs)
    This class sets the intercept to 0 by default, since usually we include it
    in X.
    """

    def __init__(self, penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None):
        #if not "fit_intercept" in kwargs:
        #    kwargs['fit_intercept'] = False
        self.fit_intercept = False
        super(LogisticRegression, self)\
                .__init__(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs, l1_ratio=l1_ratio)

    def fit(self, X, y, n_jobs=1):
        self = super(LogisticRegression, self).fit(X, y, n_jobs)

        beta_hat = self.intercept_.tolist() + self.coef_[0].tolist()

        # compute the p-values
        n = X.shape[0]
        
        # add ones column
        X1 = np.column_stack((np.ones(n), X))
        
        # standard deviation of the noise.
        sigma_hat = np.sqrt(np.sum(np.square(y - X1@beta_hat)) / (n - X1.shape[1]))
        
        # estimate the covariance matrix for beta 
        beta_cov = np.linalg.inv(X1.T@X1)
        self.se = sigma_hat * np.sqrt(np.diagonal(beta_cov))
        
        # let's test method 2 to see if we get the same
        # Initiate matrix of 0's, fill diagonal with each predicted observation's variance
        #V = np.diagflat(np.product(y, axis=1))

        #covLogit = np.linalg.inv(np.dot(np.dot(X1.T, V), X1))
        #print("Covariance matrix: ", covLogit)

        # Standard errors
        #print("Standard errors: ", np.sqrt(np.diag(covLogit)))
        #input(self.se)

        # the t-test statistic for each variable from the formula from above figure
        self.t_vals = beta_hat / self.se
        # compute 2-sided p-values.
        self.p_vals = t.sf(np.abs(self.t_vals), n-X1.shape[1])*2
        return self