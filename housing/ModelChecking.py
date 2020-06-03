import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from housing.ML import ML_Model
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, BayesianRidge

def statistical_plots(fitted_y, y):
    #Predicted vs Actual
    plot_1 = sns.scatterplot(fitted_y,y)
    plot_1.set_title('Actual vs Predicted')
    plot_1.set_xlabel('Actual values')
    plot_1.set_ylabel('Predicted');

    
    #Residual vs Fitted
    plot_2 = plt.figure()
    plot_2 = sns.residplot(fitted_y, y,
                            scatter_kws={'alpha': 0.5},
                            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    plot_2.set_title('Residuals vs Fitted')
    plot_2.set_xlabel('Fitted values')
    plot_2.set_ylabel('Residuals');

    #QQPlot
    res  = pd.DataFrame(y - fitted_y)
    plot_3 = sm.qqplot(res, stats.t, distargs=(4,), line='s');
    


def ttest(model, X):
    '''
    model refers to ML_Model
    X refers X_train
    '''
    ML_model = model.get_model()
    if (isinstance(ML_model, Lasso) or isinstance(ML_model, BayesianRidge)):
        coeffs = np.append(ML_model.intercept_, ML_model.coef_)
    elif (isinstance(ML_model, RandomForestRegressor)):
        coeffs = ML_model.estimators_
    else:
        return "Wrong ML Model"
    MSE = model.get_MSE()
    newX = np.append(np.ones((len(X),1)), X, axis=1)

    var = np.linalg.inv(np.dot(newX.T,newX))
    standard_error = np.sqrt(MSE*(var.diagonal()))
    test_statistic = coeffs / standard_error

    p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-1))) for i in test_statistic]

    standard_error = np.round(standard_error, 3)
    test_statistic = np.round(test_statistic, 3)
    p_values = np.round(p_values, 3)
    coeffs = np.round(coeffs, 3)

    ttestDF = pd.DataFrame({'Coefficients': coeffs})
    ttestDF['Standard Errors'] = standard_error
    ttestDF['t values'] = test_statistic
    ttestDF['Probabilities'] = p_values
    return ttestDF

def AIC_BIC(fitted_y, y, k):
    '''
    k refers to k number of variables in model
    '''
    EPSILON = 1e-4 #if SSE is very small
    resid = y - fitted_y
    SSE = sum(resid**2) + EPSILON
    n = len(fitted_y)
    AIC = 2*k - 2*np.log(SSE)
    BIC = n*np.log(SSE/n) + k*np.log(n)
    return "AIC: "+ str(AIC) + " BIC: " + str(BIC)