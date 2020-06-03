import numpy as np
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score, mean_absolute_error

class ML_Model:
    def __init__(self,model, X, y, split_fn):
        ''' Args
        model - the machine learning model from sklearn to fit. 
                u can use a custom class but make sure if has the methods similar to sklearn
        df - the full dataset
        split_fn - the function used to split the df into X_train, X_test, y_train, y_test. 
                   Should be called as such: split_fn(X, y)
        '''
        
        self.model = model
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test =  split_fn(X,y)
        self.metrics = {}
        self.preds = None

    def fit(self):
        self.model.fit(self.X_train, self.y_train)

        preds = self.model.predict(self.X_train)
        print("Adding fitted regression metrics ...")
        self.metrics["fitted_metrics"] = {
            "MSE" : mean_squared_error(self.y_train, preds),
            "Explained Variance" : explained_variance_score(self.y_train, preds),
            "R^2" : r2_score(self.y_train, preds),
            "MAE" : mean_absolute_error(self.y_train, preds),
        }
        self.metrics["fitted_metrics"]["RMSE"] = np.sqrt([self.metrics["fitted_metrics"]["MSE"]])
        self.metrics["fitted_metrics"]["MSPE"] = self.metrics["fitted_metrics"]["MSE"] / (np.average(self.y_train)**2) * 100

    def predict_test(self, ):
        self.preds = self.model.predict(self.X_test)
        
        print("Adding OOB regression metrics ...")
        self.metrics["test_metrics"] = {
            "MSE" : mean_squared_error(self.y_test, self.preds),
            "Explained Variance" : explained_variance_score(self.y_test, self.preds),
            "R^2" : r2_score(self.y_test, self.preds),
            "MAE" : mean_absolute_error(self.y_test, self.preds),
        }
        self.metrics["test_metrics"]["RMSE"] = np.sqrt([self.metrics["test_metrics"]["MSE"]])
        self.metrics["test_metrics"]["MSPE"] = self.metrics["test_metrics"]["MSE"] / (np.average(self.y_train)**2) * 100
        return self.preds
    
    def predict(self, test):
        return self.model.predict(test)
        
    def rolling_predict(self, rolling_month):
        roll_df = pd.concat([self.X,self.y], axis=1)
        max_month = np.amax(roll_df['month'])
        temp_mth = np.amin(roll_df['month'])
        
        MSEs = []
        MAEs = []
        ExplainVars = []
        R_squareds = []
        predicted_vals =[]
        
        while temp_mth < (max_month-rolling_month):
            X_train = roll_df[(roll_df['month'] > temp_mth) & (roll_df['month'] <= temp_mth+rolling_month)].drop('resale_price',axis=1)
            X_test = roll_df[(roll_df['month'] > temp_mth+rolling_month) & (roll_df['month'] <= temp_mth+2*rolling_month)].drop('resale_price', axis=1)
            y_train = roll_df[(roll_df['month'] > temp_mth) & (roll_df['month'] <= temp_mth+rolling_month)]['resale_price']
            y_test = roll_df[(roll_df['month'] > temp_mth+rolling_month) & (roll_df['month'] <= temp_mth+2*rolling_month)]['resale_price']
            
            print(X_train.shape, y_train.shape)

            
            if X_train.shape[0] == y_train.shape[0] and y_train.shape[0] > 1000:
                self.model.fit(X_train,y_train)
                predicted_vals.append(self.model.predict(X_test))
                
                MAEs.append(mean_absolute_error(predicted_vals[-1], y_test))
                MSEs.append(mean_squared_error(predicted_vals[-1], y_test))
                ExplainVars.append(explained_variance_score(predicted_vals[-1], y_test))
                R_squareds.append(r2_score(predicted_vals[-1], y_test))
            temp_mth += rolling_month
        
        print("Adding some metrics: mse, R_squared, Explained variance")
        self.metrics["(Rolling mean) MAE"] = np.mean(MAEs)
        self.metrics["(Rolling mean) MSE"] = np.mean(MSEs)
        self.metrics["(Rolling mean) R^2"] = np.mean(ExplainVars)
        self.metrics["(Rolling mean) Explained Variance"] = np.mean(R_squareds)
        
        self.rolling_predicted =  predicted_vals
        
    def add_metric(self, name, fn):
        '''
        Adds a metric not already defined. Metric should take in self.preds and self.y_test
        '''
        self.metrics[name] = fn(self.preds, self.y_test)
        
    def get_metrics(self,):
        return self.metrics
    
    def get_predicted(self,):
        if self.preds == None:
            raise Exception("prediction is not performed yet")
        return self.preds

    def get_model(self,):
        return self.model  
    

class Plotting:
    def __init__(self,):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(11,7))
    
    def residual_plot(self, preds, actual):
        plt.plot(actual - preds)
    
    def viz_predicted(self, preds, actual):
        plt.plot(preds, label='predicted', color='tab:orange')
        plt.plot(actual, label='actual', color='tab:blue')