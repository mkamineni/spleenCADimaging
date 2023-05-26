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
        super(LogisticRegression, self)\
                .__init__(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs, l1_ratio=l1_ratio)

    def fit(self, X, y):
        '''
        Fits the logistic regresison model but also calculates the standard errors and p-values for logistic regression coefficients.
        SE for logistic regression coefficients can be estimated as the sqrt of the diagnoal of the inverse of the fisher information matrix
        '''
        self = super(LogisticRegression, self).fit(X, y)
        denom = (2.0*(1.0+np.cosh(self.decision_function(X))))
        denom = np.tile(denom,(X.shape[1],1)).T
        F_ij = np.dot((X/denom).T,X) ## Fisher Information Matrix
        Cramer_Rao = np.linalg.inv(F_ij) ## Inverse Information Matrix
        
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.coef_[0]/sigma_estimates # z-score for eaach model coefficient
        p_values = [stats.norm.sf(abs(x))*2 for x in z_scores] ### two tailed test for p-values
        self.se = sigma_estimates
        self.z_scores = z_scores
        self.p_values = p_values
        return self