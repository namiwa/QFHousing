import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from housing.ML import ML_Model
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, BayesianRidge
import random

class statplots:
    def __init__(self, fitted_y, test_y, index, model, X):
        self.fitted_y = fitted_y
        self.test_y = test_y
        self.index = index
        self.model = model
        self.X = X
        df = pd.DataFrame(fitted_y, index)
        df['test_y'] = test_y
        df.columns = ['fitted_y','test_y']
        df = df.reset_index().drop(columns=['index'])
        df['residuals'] = df['test_y'] - df['fitted_y']
        self.df = df
        self.xTransposedDotX = None
        self.hat_matrix = None
        print('statsplot for this model has been loaded!')

    def predictedVsActual(self, numOfPlots, numOfPoints):
        '''
        numOfPlots refers to the number of subplots user desires for
        numOfPoints refers to number of datapoints in each plot. Recommened to
            be less than couple hundred for easier visualisation
        '''
        
        if (len(self.fitted_y)/numOfPlots < numOfPoints):
            return "Error: Too many plots or too many points!!!"
        else:
            plotrange = len(self.fitted_y)//numOfPlots
        for i in range(numOfPlots):
            startpoint = random.randint(i*plotrange, (i+1)*plotrange - numOfPoints)
            endpoint = startpoint + numOfPoints
            plot_df = self.df.iloc[startpoint:endpoint].reset_index().drop(columns=
                ['index','residuals'])
            plot = plot_df.plot(colormap = 'Dark2')
            plot.set_xlabel("Transaction")
            plot.set_ylabel("Prices")    
    
    def residualPlot(self, *args):
        '''
        if no number passed (left empty) then assumed all points to be plotted else 
        plot a random range of number of datapoints desired
        '''
        plotrange = len(self.fitted_y)
        if (args == ()):
            startpoint = 0
            endpoint = plotrange
        else:
            numOfPoints = args[0]
            startpoint = random.randint(0, plotrange - numOfPoints)
            endpoint = startpoint + numOfPoints
        sns.scatterplot(data = 
            self.df.drop(columns=['fitted_y','test_y'])[startpoint:endpoint])
        plt.xlabel("Transactions")
        plt.ylabel("Prices ($)")
        plt.title("Residual Plot")
        
    def residualVsFitted(self, *args):
        '''
        if no number passed (left empty) then assumed all points to be plotted else 
        plot a random range of number of datapoints desired
        '''
        plotrange = len(self.fitted_y)
        if (args == ()):
            startpoint = 0
            endpoint = plotrange
            #plot = sns.residplot(self.fitted_y, self.test_y,
            #                        scatter_kws={'alpha': 0.5},
            #                        line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
            #plot.set_title('Residuals vs Fitted')
            #plot.set_xlabel('Fitted values')
            #plot.set_ylabel('Residuals');
        else:
            numOfPoints = args[0]
            startpoint = random.randint(0, plotrange - numOfPoints)
            endpoint = startpoint + numOfPoints
        sns.scatterplot(self.df['fitted_y'][startpoint:endpoint], 
            self.df['residuals'][startpoint:endpoint])
        plt.xlabel("Fitted Prices ($)")
        plt.ylabel("Residual Prices ($)")
        plt.title("Residual vs Fitted")



    def qqplot(self):
        residual = self.test_y - self.fitted_y
        if (self.xTransposedDotX == None):
            self.hat_matrix_calc()
        MSE = self.model.get_MSE()
        residual_var = MSE*(np.ones((len(self.X),1)) - self.hat_matrix.diagonal())
        studentised_residual = residual / residual_var
        plot = sm.qqplot(studentised_residual, dist=stats.t, distargs=(4,), line='s');
    
    def hat_matrix_calc(self):
        '''
        helper function to get hat_matrix
        '''
        newX = np.append(np.ones((len(self.X),1)), self.X, axis=1)
        self.xTransposedDotX = np.linalg.inv(np.dot(newX.T,newX))
        self.hat_matrix = np.dot(np.dot(newX, self.xTransposedDotX), newX.T)

    def ttest(self):
        ML_model = self.model.get_model()
        if (isinstance(ML_model, Lasso) or isinstance(ML_model, BayesianRidge)):
            coeffs = np.append(ML_model.intercept_, ML_model.coef_)
        elif (isinstance(ML_model, RandomForestRegressor)):
            coeffs = ML_model.estimators_
        else:
            return "Wrong ML Model"
        MSE = self.model.get_MSE()
        if (self.xTransposedDotX == None):
            self.hat_matrix_calc()
        standard_error = np.sqrt(MSE*(self.xTransposedDotX.diagonal()))
        test_statistic = coeffs / standard_error

        p_values =[2*(1-stats.t.cdf(np.abs(i),(len(self.X)))) for i in test_statistic]

        standard_error = np.round(standard_error, 3)
        test_statistic = np.round(test_statistic, 3)
        p_values = np.round(p_values, 3)
        coeffs = np.round(coeffs, 3)

        ttestDF = pd.DataFrame({'Coefficients': coeffs})
        ttestDF['Standard Errors'] = standard_error
        ttestDF['t values'] = test_statistic
        ttestDF['Probabilities'] = p_values
        return ttestDF

    def AIC_BIC(self):
        '''
        k refers to k number of variables in model
        '''
        k = len(self.X.iloc[0])
        EPSILON = 1e-4 #if SSE is very small
        resid = self.test_y - self.fitted_y
        SSE = sum(resid**2) + EPSILON
        n = len(self.fitted_y)
        AIC = 2*k - 2*np.log(SSE)
        BIC = n*np.log(SSE/n) + k*np.log(n)
        # "AIC: "+ str(AIC) + " BIC: " + str(BIC)
        return AIC, BIC