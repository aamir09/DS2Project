# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import  StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from models import linearRegression as lr
from models import neuralNetwork as nn
import tensorflow as tf

#Creating Pipeline Classes


#Dropping Columns
class dropColumns(BaseEstimator,TransformerMixin):
    def __init__(self,cols):
        #Calling the constructor method of both base classes 
        super().__init__()

        #Creating a variable to store columns to be dropped
        self.dropCols=cols

    def fit(self, X,y=None):
        #Pipelines fit  function
        return self
    
    def transform(self,X):
        print(f'Dropping Columns : {self.dropCols}')
        X=X.drop(columns=list(self.dropCols),axis=1)
        return X
    
    def fit_transform(self,X,y=None):
        print(f'Dropping Columns : {self.dropCols}')
        X=X.drop(columns=list(self.dropCols),axis=1)
        return X

#Imputation 

class Imputer(BaseEstimator, TransformerMixin):
    def __init__(self,cols):
        super().__init__()

        self.cols=cols
        self.imputer=None
    
    def fit(self,X,y=None):
        self.imputer=IterativeImputer(random_state=2022)
        return self

    def transform(self,X):
        XImputed=self.imputer.transform(X[self.cols])
        XNew=pd.DataFrame(XImputed,columns=self.cols)
        return XNew
    def fit_transform(self,X,y=None):
        self.imputer=IterativeImputer(random_state=2022)
        XImputed=self.imputer.fit_transform(X[self.cols])
        XNew=pd.DataFrame(XImputed,columns=self.cols)
        return XNew




class standardize(BaseEstimator, TransformerMixin):

    def __init__(self,cols):
        super().__init__()
        self.cols=cols
    
    def fit(self,X,y=None):
        return self

    def transform(self,X):

        scaler=StandardScaler()
        XScaled=scaler.fit_transform(X[self.cols])
        return XScaled

class encoder(BaseEstimator, TransformerMixin):

    def __init__(self,cols):
        super().__init__()
        self.cols=cols
    def fit(self,X,y=None):
        return self

    def transform(self,X):
        XNew=pd.DataFrame()
        for col in self.cols:
            dummies=pd.get_dummies(X[col])
            X=X.drop(col,axis=1)
            XNew=pd.concat([XNew,dummies],axis=1)
        return XNew.values


class categoricalImputer(BaseEstimator, TransformerMixin):

    def __init__(self,cols):
        super().__init__()

        self.cols=cols

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        imputer =SimpleImputer(strategy='most_frequent')
        XImputed=imputer.fit_transform(X[self.cols])
        return pd.DataFrame(XImputed,columns=self.cols)

#Nan Replacer
class replaceWithNan(BaseEstimator, TransformerMixin):

    def __init__(self):
        super().__init__()

    def fit(self,X,y=None):
        return self

    def fit_transform(self,X,y=None):
        return X.replace(9,np.nan)
    
    def transform(self,X):
        return X.replace(9,np.nan)



class trainModels(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        self.linearModel=None
        self.neuralNetwork=None

    def fit(self,X,y=None):
        return self
    
    def fit_transform(self,X,y):
        #Train Linear Model
        self.linearModel=lr.trainModel(X,y)

        #Train Neural Network
        optimizer=tf.keras.optimizers.Adam(1e-3)
        loss='mse'
        metrics=['mse']
        self.neuralNetwork=nn.trainNn(X,y,optimizer=optimizer,loss=loss,metrics=metrics)

        return self.linearModel,self.neuralNetwork

    def transform(self,X):
        return X